include("FLOWMath.jl")
using Main.FLOWMath: Akima
using LinearAlgebra
using ProgressMeter
using Parameters
using QuantEcon: rouwenhorst, tauchen, gridmake, MarkovChain, stationary_distributions
using Plots
using PrettyTables
using Roots
using Optim
using Calculus
using Distributions
using SparseArrays
using BSON: @save, @load
using UnicodePlots: spy

function para_func(;
    ψ_B::Real = 0.80,           # bank's survival rate
    β_H::Real = 0.96,           # discount factor (households)
    β_B::Real = 0.90,           # discount factor (banks)
    ξ::Real = 0.40,             # garnishment rate
    σ::Real = 2.0,              # CRRA coefficient
    z::Real = 1.0,              # aggregate uncertainty
    r_d::Real = 0.02,           # deposit rate
    λ::Real = 0.00,             # the multiplier of incentive constraint
    θ::Real = 0.20,             # the diverted fraction
    e_ρ::Real = 0.95,           # AR(1) of earnings shock
    e_σ::Real = 0.10,           # s.d. of earnings shock
    e_size::Integer = 15,       # no. of earnings shock
    ν_size::Integer = 2,        # no. of preference shock
    ν_s::Real = 0.80,           # scale of patience
    ν_p::Real = 0.10,           # probability of patience
    a_min::Real = -4.5,         # min of asset holding
    a_max::Real = 350.0,        # max of asset holding
    a_size_neg::Integer = 451,  # number of the grid of negative asset holding for VFI
    a_size_pos::Integer = 51,   # number of the grid of positive asset holding for VFI
    a_degree::Integer = 3,      # curvature of the positive asset gridpoints
    μ_scale::Integer = 5        # scale governing the number of grids in computing density
    )
    #-----------------------------------------------------#
    # contruct an immutable object containg all paramters #
    #-----------------------------------------------------#

    # persistent shock
    e_M = tauchen(e_size, e_ρ, e_σ, 0.0, 8)
    e_Γ = e_M.p
    e_grid = collect(e_M.state_values)

    # preference schock
    ν_grid = [ν_s, 1.0]
    ν_Γ = repeat([ν_p 1.0-ν_p], ν_size, 1)

    # idiosyncratic transition matrix
    x_Γ = kron(ν_Γ, e_Γ)
    x_grid = gridmake(e_grid, ν_grid)
    x_ind = gridmake(1:e_size, 1:ν_size)
    x_size = e_size*ν_size

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length = a_size_neg))
    a_grid_pos = ((range(0.0, stop = a_size_pos-1, length = a_size_pos)/(a_size_pos-1)).^a_degree)*a_max
    a_grid = cat(a_grid_neg, a_grid_pos[2:end], dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero,a_grid)[]

    # asset holding grid for μ
    a_size_neg_μ = convert(Int, a_size_neg*μ_scale)
    a_grid_neg_μ = collect(range(a_min, 0.0, length = a_size_neg_μ))
    a_size_pos_μ = convert(Int, a_size_pos*μ_scale)
    a_grid_pos_μ = collect(range(0.0, a_max, length = a_size_pos_μ))
    a_grid_μ = cat(a_grid_neg_μ, a_grid_pos_μ[2:end], dims = 1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(iszero,a_grid_μ)[]

    # auxiliary indicies for density
    μ_ind = gridmake(1:a_size_μ, 1:e_size, 1:ν_size)

    # solve the steady state of ω and θ to match targeted parameters
    # load equilibrium objects
    α = (β_B*(1.0-ψ_B)*(1.0+r_d)) / ((1.0-λ)-β_B*ψ_B*(1.0+r_d))
    Λ = β_B*(1.0-ψ_B+ψ_B*α)
    r_ld = λ*θ/Λ
    LR = α/θ
    AD = LR/(LR-1)
    ω = ((r_ld*(α/θ)+(1+r_d))^(-1)-(1-ψ_B))/ψ_B

    # return values
    return (ψ_B = ψ_B, β_H = β_H, β_B = β_B, ξ = ξ, σ = σ, z = z, AD = AD,
            LR = LR, r_ld = r_ld, r_d = r_d, θ = θ, λ = λ, α = α, Λ = Λ, ω = ω,
            a_grid = a_grid, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos,
            a_size = a_size, a_size_neg = a_size_neg, a_size_pos = a_size_pos,
            a_ind_zero = a_ind_zero,
            a_grid_μ = a_grid_μ, a_grid_neg_μ = a_grid_neg_μ, a_grid_pos_μ = a_grid_pos_μ,
            a_size_μ = a_size_μ, a_size_neg_μ = a_size_neg_μ, a_size_pos_μ = a_size_pos_μ,
            a_ind_zero_μ = a_ind_zero_μ, μ_scale = μ_scale, μ_ind = μ_ind,
            e_ρ = e_ρ, e_σ = e_σ, e_Γ = e_Γ, e_grid = e_grid, e_size = e_size,
            ν_p = ν_p, ν_Γ = ν_Γ, ν_grid = ν_grid, ν_size = ν_size,
            x_Γ = x_Γ, x_grid = x_grid, x_size = x_size, x_ind = x_ind)
end

function para_func_MIT(
    parameters::NamedTuple;
    ρ_z::Real = 0.90,           # AR(1) coefficient of aggregate shock
    σ_z::Real = 0.01,           # s.d. of aggregate shock
    T_size::Integer = 250       # time periods
    )

    @unpack e_size, e_grid, e_ρ, e_σ = parameters

    # aggregate shock
    z_path = zeros(T_size)
    z_path[1] = -σ_z
    for T_ind in 2:T_size
        z_path[T_ind] = ρ_z*z_path[T_ind-1]
    end

    # time-varying volatility
    e_σ_z = zeros(T_size)
    for T_ind in 1:T_size
        e_σ_z[T_ind] = e_σ
        # e_σ_z[T_ind] = e_σ*(1.0 + abs(z_path[T_ind])^(1/10))
        # e_σ_z[T_ind] = e_σ*(1.0 + abs(z_path[T_ind])*20)
    end

    # Tauchen method of fixed grids
    function tauchen_Γ_func(
        size::Integer,
        grid::Array{Float64,1},
        ρ::Real,
        σ::Real
        )

        Γ = zeros(size,size)
        d = grid[2]-grid[1]
        for i in 1:size
            dist = Normal(ρ*e_grid[i],σ)
            for j in 1:size
                if j == 1
                    prob = cdf(dist, grid[j]+d/2)
                elseif j == size
                    prob = 1.0 - cdf(dist, grid[j]-d/2)
                else
                    prob = cdf(dist, grid[j]+d/2) - cdf(dist, grid[j]-d/2)
                end
                Γ[i,j] = prob
            end
        end
        return Γ
    end

    # time-varying uncertainty
    e_Γ_z = zeros(e_size,e_size,T_size)
    for T_ind in 1:T_size
        # e_Γ_z[:,:,T_ind] = tauchen_Γ_func(e_size, e_grid, e_ρ, e_σ_z[T_ind])
        e_Γ_z[:,:,T_ind] = tauchen_Γ_func(e_size, e_grid, e_ρ, e_σ)
    end

    # return values
    return (T_size = T_size, z_path = z_path, e_σ_z = e_σ_z, e_Γ_z = e_Γ_z)
end

mutable struct mut_var_MIT
    q::Array{Float64,4}
    prob_default::Array{Float64,4}
    V::Array{Float64,4}
    V_repay::Array{Float64,4}
    V_default::Array{Float64,4}
    policy_a::Array{Float64,4}
    # policy_a_matrix::Array{Float64,5}
    policy_d::Array{Float64,4}
    policy_d_matrix::Array{Float64,4}
    μ::Array{Float64,4}
    LN_guess::Array{Float64,1}
    aggregate_var::Array{Float64,2}
    prices::Array{Float64,2}
end

function var_func_MIT(
    parameters::NamedTuple,
    parameters_MIT::NamedTuple
    )
    #------------------------------------------------------------#
    # construct a mutable object containing endogenous variables #
    #------------------------------------------------------------#

    # unpack parameters
    @unpack a_grid, a_size, a_size_pos, a_ind_zero = parameters
    @unpack a_size_μ = parameters
    @unpack x_size, ν_size, e_size, e_grid = parameters
    @unpack σ, r_d, r_ld = parameters
    @unpack T_size, z_path = parameters_MIT

    # load equilibrium objects
    @load "optimal_values.bson" Vss V_repayss V_defautlss qss μss Lss Dss Nss αss λss Λss

    # define pricing related variables
    q = ones(a_size, e_size, ν_size, T_size)
    q[:,:,:,end] = qss

    # define default probability
    prob_default = zeros(a_size, e_size, ν_size, T_size)

    # define value functions
    V = zeros(a_size, e_size, ν_size, T_size)
    V[:,:,:,end] = Vss
    V_repay = zeros(a_size, e_size, ν_size, T_size)
    V_repay[:,:,:,end] = V_repayss
    V_default = zeros(a_size, e_size, ν_size, T_size)
    V_default[:,:,:,end] = V_defautlss

    # define policy functions
    policy_a = zeros(a_size, e_size, ν_size, T_size)
    # policy_a_matrix = zeros(a_size_μ, a_size_μ, e_size, ν_size, T_size)
    policy_d = zeros(a_size, e_size, ν_size, T_size)
    policy_d_matrix = zeros(a_size_μ, e_size, ν_size, T_size)

    # define the type distribution and its transition matrix
    μ_size = x_size*a_size_μ
    μ = zeros(a_size_μ, e_size, ν_size, T_size)
    μ[:,:,:,1] = μss

    # load initial guess
    # @load "initial_guess.bson" LN_guess
    LN_guess = ones(T_size) .* (Lss/Nss)
    # LN_guess = (1.0 .- z_path.^5)*(Lss/Nss)
    # LN_guess = (1.0 .+ (-z_path).^(1/2.0))*(Lss/Nss)

    # define aggregate objects
    aggregate_var = zeros(5, T_size)
    aggregate_var[1,end] = Lss
    aggregate_var[2,end] = Dss
    aggregate_var[3,end] = Nss
    aggregate_var[3,1] = 0.99*Nss
    aggregate_var[4,end] = Lss/Dss
    aggregate_var[5,:] = LN_guess

    # equilibrium prices
    prices = zeros(4, T_size) # rows: α, Λ, λ, r_ld

    # return outputs
    variables = mut_var_MIT(q, prob_default, V, V_repay, V_default,
                            policy_a, # policy_a_matrix,
                            policy_d, policy_d_matrix,
                            μ, LN_guess, aggregate_var, prices)
    return variables
end

function updating_prics_MIT!(
    variables::mut_var_MIT,
    parameters::NamedTuple
    )

    @unpack θ, β_B, ψ_B, α, r_d = parameters

    # load equilibrium objects
    @load "optimal_values.bson" Vss V_repayss V_defautlss qss μss Lss Dss Nss αss λss Λss

    # α, Λ, λ, r_ld
    variables.prices[1,:] = θ*variables.LN_guess
    # variables.prices[1,:] .= (β_B*(1.0-ψ_B)*(1.0+r_d)) / ((1.0-λss)-β_B*ψ_B*(1.0+r_d))
    variables.prices[2,:] = β_B*(1.0 .- ψ_B .+ ψ_B*variables.prices[1,:])
    variables.prices[3,:] = [(1.0 .- ((variables.prices[2,2:end]*(1+r_d)) ./ variables.prices[1,1:(end-1)])); λss]
    variables.prices[4,:] = θ*(variables.prices[3,:] ./ [variables.prices[2,2:end]; Λss])
end

function u_func(c::Real, σ::Real)
    #--------------------------------------------------------------#
    # compute utility of CRRA utility function with coefficient σ. #
    #--------------------------------------------------------------#
    if c > 0
        return σ == 1 ? log(c) : 1 / ((1-σ)*c^(σ-1))
    else
        return -10^15
    end
end

function value_price_func_MIT!(
    variables::mut_var_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple
    )

    @unpack a_grid, a_grid_pos, a_grid_neg, a_ind_zero = parameters
    @unpack a_size, a_size_pos = parameters
    @unpack x_grid, x_size, x_ind, x_Γ = parameters
    @unpack β_H, σ, ξ, r_d, r_ld = parameters
    @unpack e_size, e_grid, e_ρ, e_σ = parameters
    @unpack ν_p, ν_Γ, ν_size = parameters

    @unpack T_size, z_path, e_σ_z, e_Γ_z = parameters_MIT

    @showprogress 1 "Solving household's problem backward..." for T_ind in (T_size-1):(-1):1

        z = z_path[T_ind]
        e_σ = e_σ_z[T_ind]
        e_Γ = e_Γ_z[:,:,T_ind]
        V_p = variables.V[:,:,:,(T_ind+1)]

        Threads.@threads for x_i in 1:x_size

            e, ν = x_grid[x_i,:]
            e_i, ν_i = x_ind[x_i,:]

            V_expt_p = (ν_p*V_p[:,:,1] + (1-ν_p)*V_p[:,:,2])*e_Γ[e_i,:]
            V_hat = (ν*β_H)*V_expt_p
            V_hat_itp = Akima(a_grid, V_hat)

            variables.V_default[:,e_i,ν_i,T_ind] .= u_func((1-ξ)*exp(z+e), σ) + V_hat[a_ind_zero]
            # variables.V_default[:,e_i,ν_i,T_ind] .= u_func((1-ξ)*exp(e), σ) + V_hat[a_ind_zero]

            q = (variables.q[:,e_i,ν_i,(T_ind+1)] .* (1.0+r_d+variables.prices[4,(T_ind+1)])) ./ (1.0+r_d+variables.prices[4,T_ind])
            # q = variables.q[:,e_i,ν_i,(T_ind+1)]
            qa = q .* a_grid
            qa_itp = Akima(a_grid, qa)

            for a_i in 1:a_size
                a = a_grid[a_i]
                CoH = exp(z+e) + (1.0+r_d*(a>0.0))*a
                # CoH = exp(e) + (1+r_d*(a>0))*a

                # identify the optimal regions with discrete gridpoints
                V_all = u_func.(CoH .- qa, σ) .+ V_hat
                V_max_ind = argmax(V_all)

                # solve it with interpolation method
                object_good(a_p) = -(u_func(CoH - qa_itp(a_p[1]), σ) + V_hat_itp(a_p[1]))
                function gradient_good!(G, a_p)
                    G[1] = derivative(object_good, a_p[1])
                end

                if a_grid[V_max_ind] >= CoH
                    initial_good = CoH - 10^(-6)
                elseif a_grid[V_max_ind] <= a_grid[1]
                    initial_good = a_grid[1] + 10^(-6)
                else
                    initial_good = a_grid[V_max_ind]
                end
                res_good = optimize(object_good, gradient_good!, [a_grid[1]], [CoH], [initial_good], Fminbox(GradientDescent()))

                # record results
                variables.V_repay[a_i,e_i,ν_i,T_ind] = -Optim.minimum(res_good)
                if variables.V_default[a_i,e_i,ν_i,T_ind] > variables.V_repay[a_i,e_i,ν_i,T_ind]
                    variables.V[a_i,e_i,ν_i,T_ind] = variables.V_default[a_i,e_i,ν_i,T_ind]
                    variables.policy_a[a_i,e_i,ν_i,T_ind] = 0.0
                    variables.policy_d[a_i,e_i,ν_i,T_ind] = 1.0
                else
                    variables.V[a_i,e_i,ν_i,T_ind] = variables.V_repay[a_i,e_i,ν_i,T_ind]
                    variables.policy_a[a_i,e_i,ν_i,T_ind] = Optim.minimizer(res_good)[]
                    variables.policy_d[a_i,e_i,ν_i,T_ind] = 0.0
                end
            end
        end

        a_grid_neg_nozero = a_grid_neg[1:(end-1)]
        a_size_neg_nozero = length(a_grid_neg_nozero)
        q_update = ones(a_size_neg_nozero, e_size, ν_size) ./ (1.0+r_d+variables.prices[4,T_ind])

        Threads.@threads for a_p_i in 1:a_size_neg_nozero

            # impatient household
            V_diff_1 = variables.V_repay[a_p_i,:,1,T_ind] .- variables.V_default[a_p_i,:,1,T_ind]
            if all(V_diff_1 .> 0.0)
                e_p_thres_1 = -Inf
            elseif all(V_diff_1 .< 0.0)
                e_p_thres_1 = Inf
            else
                e_p_lower_1 = e_grid[findall(V_diff_1 .<= 0.0)[end]]
                e_p_upper_1 = e_grid[findall(V_diff_1 .>= 0.0)[1]]
                V_diff_1_itp = Akima(e_grid, V_diff_1)
                object_V_diff_1(e_p) = V_diff_1_itp(e_p)
                e_p_thres_1 = find_zero(object_V_diff_1, (e_p_lower_1, e_p_upper_1), Bisection())
            end

            # patient household
            V_diff_2 = variables.V_repay[a_p_i,:,2,T_ind] .- variables.V_default[a_p_i,:,2,T_ind]
            if all(V_diff_2 .> 0.0)
                e_p_thres_2 = -Inf
            elseif all(V_diff_2 .< 0.0)
                e_p_thres_2 = Inf
            else
                e_p_lower_2 = e_grid[findall(V_diff_2 .<= 0.0)[end]]
                e_p_upper_2 = e_grid[findall(V_diff_2 .>= 0.0)[1]]
                V_diff_2_itp = Akima(e_grid, V_diff_2)
                object_V_diff_2(e_p) = V_diff_2_itp(e_p)
                e_p_thres_2 = find_zero(object_V_diff_2, (e_p_lower_2, e_p_upper_2), Bisection())
            end

            for e_i in 1:e_size
                e = e_grid[e_i]
                dist = Normal(0.0,1.0)

                default_prob_1 = cdf(dist, (e_p_thres_1-e_ρ*e)/e_σ)
                default_prob_2 = cdf(dist, (e_p_thres_2-e_ρ*e)/e_σ)
                # default_prob_1 = cdf(dist, (e_p_thres_1-e_ρ*e)/(e_σ))
                # default_prob_2 = cdf(dist, (e_p_thres_2-e_ρ*e)/(e_σ))
                repay_prob = ν_p*(1.0-default_prob_1) + (1.0-ν_p)*(1.0-default_prob_2)
                variables.prob_default[a_p_i,e_i,:,T_ind] .= 1.0 - repay_prob
                q_update[a_p_i,e_i,:] .= repay_prob / (1.0+r_d+variables.prices[4,T_ind])
            end
        end
        variables.q[1:a_size_neg_nozero,:,:,T_ind] = q_update
    end
end

#=
function value_func_MIT!(
    T_ind::Integer,
    variables::mut_var_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple
    )

    @unpack a_grid, a_grid_pos, a_ind_zero = parameters
    @unpack a_size, a_size_pos = parameters
    @unpack x_grid, x_size, x_ind, x_Γ = parameters
    @unpack e_grid, ν_p = parameters
    @unpack β_H, σ, ξ, r_d, r_ld = parameters

    @unpack z, e_Γ_z = parameters_MIT

    z = z[T_ind]
    e_Γ = e_Γ_z[:,:,T_ind]
    V_p = variables.V[:,:,:,(T_ind+1)]

    for x_i in 1:x_size

        e, ν = x_grid[x_i,:]
        e_i, ν_i = x_ind[x_i,:]

        V_expt_p = (ν_p*V_p[:,:,1] + (1-ν_p)*V_p[:,:,2])*e_Γ[e_i,:]
        V_hat = (ν*β_H)*V_expt_p
        V_hat_itp = Akima(a_grid, V_hat)

        variables.V_default[:,e_i,ν_i,T_ind] .= u_func((1-ξ)*exp(z+e), σ) + V_hat[a_ind_zero]

        q = variables.q[:,e_i,ν_i,(T_ind+1)]
        qa = q .* a_grid
        qa_itp = Akima(a_grid, qa)

        for a_i in 1:a_size
            a = a_grid[a_i]
            CoH = exp(z+e) + (1+r_d*(a>0))*a

            # identify the optimal regions with discrete gridpoints
            V_all = u_func.(CoH .- qa, σ) .+ V_hat
            # V_max = maximum(V_all)
            # V_max_ind = findall(V_all .== V_max)[1]
            V_max_ind = argmax(V_all)

            # solve it with interpolation method
            object_good(a_p) = -(u_func(CoH - qa_itp(a_p[1]), σ) + V_hat_itp(a_p[1]))
            function gradient_good!(G, a_p)
                G[1] = derivative(object_good, a_p[1])
            end

            if a_grid[V_max_ind] >= CoH
                initial_good = CoH - 10^(-6)
            elseif a_grid[V_max_ind] <= a_grid[1]
                initial_good = a_grid[1] + 10^(-6)
            else
                initial_good = a_grid[V_max_ind]
            end
            res_good = optimize(object_good, gradient_good!, [a_grid[1]], [CoH], [initial_good], Fminbox(GradientDescent()))

            # record results
            variables.V_repay[a_i,e_i,ν_i,T_ind] = -Optim.minimum(res_good)
            if variables.V_default[a_i,e_i,ν_i,T_ind] > variables.V_repay[a_i,e_i,ν_i,T_ind]
                variables.V[a_i,e_i,ν_i,T_ind] = variables.V_default[a_i,e_i,ν_i,T_ind]
                variables.policy_a[a_i,e_i,ν_i,T_ind] = 0.0
                variables.policy_d[a_i,e_i,ν_i,T_ind] = 1.0
            else
                variables.V[a_i,e_i,ν_i,T_ind] = variables.V_repay[a_i,e_i,ν_i,T_ind]
                variables.policy_a[a_i,e_i,ν_i,T_ind] = Optim.minimizer(res_good)[]
                variables.policy_d[a_i,e_i,ν_i,T_ind] = 0.0
            end
        end
    end
end
=#

#=
function price_func_MIT!(
    T_ind::Integer,
    variables::mut_var_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple
    )
    #----------------------------#
    # update the price schedule. #
    #----------------------------#
    @unpack ξ, r_d, r_ld = parameters
    @unpack a_size, a_grid, a_grid_neg, a_size_neg = parameters
    @unpack x_ind, x_size, x_grid, x_Γ = parameters
    @unpack e_size, e_grid, e_ρ, e_σ = parameters
    @unpack ν_p, ν_Γ, ν_size = parameters

    @unpack z, e_Γ_z = parameters_MIT

    z = z[T_ind]
    e_Γ = e_Γ_z[:,:,T_ind]

    a_grid_neg_nozero = a_grid_neg[1:(end-1)]
    a_size_neg_nozero = length(a_grid_neg_nozero)
    q_update = ones(a_size_neg_nozero, e_size, ν_size) ./ (1.0+r_d+variables.prices[4,T_ind])

    for a_p_i in 1:a_size_neg_nozero

        # impatient household
        V_diff_1 = variables.V_repay[a_p_i,:,1,T_ind+1] .- variables.V_default[a_p_i,:,1,T_ind+1]
        if all(V_diff_1 .> 0)
            e_p_thres_1 = -Inf
        elseif all(V_diff_1 .< 0)
            e_p_thres_1 = Inf
        else
            e_p_lower_1 = e_grid[findall(V_diff_1 .<= 0.0)[end]]
            e_p_upper_1 = e_grid[findall(V_diff_1 .>= 0.0)[1]]
            V_diff_1_itp = Akima(e_grid, V_diff_1)
            object_V_diff_1(e_p) = V_diff_1_itp(e_p)
            e_p_thres_1 = find_zero(object_V_diff_1, (e_p_lower_1, e_p_upper_1), Bisection())
        end

        # patient household
        V_diff_2 = variables.V_repay[a_p_i,:,2,T_ind+1] .- variables.V_default[a_p_i,:,2,T_ind+1]
        if all(V_diff_2 .> 0)
            e_p_thres_2 = -Inf
        elseif all(V_diff_2 .< 0)
            e_p_thres_2 = Inf
        else
            e_p_lower_2 = e_grid[findall(V_diff_2 .<= 0.0)[end]]
            e_p_upper_2 = e_grid[findall(V_diff_2 .>= 0.0)[1]]
            V_diff_2_itp = Akima(e_grid, V_diff_2)
            object_V_diff_2(e_p) = V_diff_2_itp(e_p)
            e_p_thres_2 = find_zero(object_V_diff_2, (e_p_lower_2, e_p_upper_2), Bisection())
        end

        for e_i in 1:e_size
            e = e_grid[e_i]
            dist = Normal(0.0,1.0)

            default_prob_1 = cdf(dist, (e_p_thres_1-e_ρ*e)/((1-z)*e_σ))
            default_prob_2 = cdf(dist, (e_p_thres_2-e_ρ*e)/((1-z)*e_σ))
            repay_prob = ν_p*(1.0-default_prob_1) + (1.0-ν_p)*(1.0-default_prob_2)
            variables.prob_default[a_p_i,e_i,:,T_ind] .= 1.0 - repay_prob
            q_update[a_p_i,e_i,:] .= repay_prob / (1.0+r_d+variables.prices[4,T_ind])
        end
    end
    variables.q[1:a_size_neg_nozero,:,:,T_ind] = q_update
end
=#

function household_func_MIT!(
    variables::mut_var_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple,
    )

    @unpack T_size = parameters_MIT

    @showprogress 1 "Solving household's problem backward..." for T_ind in (T_size-1):(-1):1

        value_price_func_MIT!(T_ind, variables, parameters, parameters_MIT)
        # update value function
        # value_func_MIT!(T_ind, variables, parameters, parameters_MIT)
        # update policy matrices
        policy_matrix_func_MIT!(T_ind, variables, parameters)
        # update price
        # price_func_MIT!(T_ind, variables, parameters, parameters_MIT)
    end
end

function policy_matrix_func_MIT!(
    variables::mut_var_MIT,
    parameters::NamedTuple
    )

    @unpack x_size, x_ind, a_grid, a_size = parameters
    @unpack a_size_μ, a_grid_μ, a_ind_zero_μ = parameters

    @unpack T_size = parameters_MIT

    for T_ind in (T_size-1):(-1):1
        for x_i in 1:x_size
            e_i, ν_i = x_ind[x_i,:]
            policy_a_itp = Akima(a_grid, variables.policy_a[:,e_i,ν_i,T_ind])

            if sum(variables.policy_d[:,e_i,ν_i,T_ind]) > 0.0
                a_ind_default = findall(isone,variables.policy_d[:,e_i,ν_i,T_ind])[end]
                a_default = a_grid[a_ind_default]
                a_repay = a_grid[a_ind_default+1]
            else
                a_default = -Inf
                a_repay = a_grid[1]
            end

            for a_i in 1:a_size_μ
                a_μ = a_grid_μ[a_i]
                # repay
                if abs(a_μ - a_repay) <= abs(a_μ - a_default)
                    a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])
                    ind_lower_a_p = findall(a_grid_μ .<= a_p)[end]
                    ind_upper_a_p = findall(a_p .<= a_grid_μ)[1]
                    if ind_lower_a_p != ind_upper_a_p
                        a_lower_a_p = a_grid_μ[ind_lower_a_p]
                        a_upper_a_p = a_grid_μ[ind_upper_a_p]
                        weight_lower = (a_upper_a_p - a_p) / (a_upper_a_p - a_lower_a_p)
                        weight_upper = (a_p - a_lower_a_p) / (a_upper_a_p - a_lower_a_p)
                        # variables.policy_a_matrix[a_i,ind_lower_a_p,e_i,ν_i,T_ind] = weight_lower
                        # variables.policy_a_matrix[a_i,ind_upper_a_p,e_i,ν_i,T_ind] = weight_upper
                    else
                        # variables.policy_a_matrix[a_i,ind_lower_a_p,e_i,ν_i,T_ind] = 1.0
                    end
                    variables.policy_d_matrix[a_i,e_i,ν_i,T_ind] = 0.0
                # default
                else
                    # variables.policy_a_matrix[a_i,a_ind_zero_μ,e_i,ν_i,T_ind] = 1.0
                    variables.policy_d_matrix[a_i,e_i,ν_i,T_ind] = 1.0
                end
            end
        end
    end
end

function density_func_MIT!(
    variables::mut_var_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple
    )

    @unpack x_size, x_ind, ν_Γ, e_size, ν_size = parameters
    @unpack a_size, a_grid, a_size_pos, a_grid_pos = parameters
    @unpack a_size_μ, a_grid_μ, a_size_pos_μ, a_grid_pos_μ, a_ind_zero_μ = parameters

    @unpack T_size, z_path, e_Γ_z = parameters_MIT

    variables.μ[:,:,:,2:end] .= 0.0

    @showprogress 1 "Solving density forward..." for T_ind in 1:(T_size-1)

        z = z_path[T_ind]
        e_Γ = e_Γ_z[:,:,(T_ind+1)]

        μ_p = similar(variables.μ[:,:,:,T_ind])
        copyto!(μ_p, variables.μ[:,:,:,T_ind])
        μ_temp = zeros(a_size_μ, e_size, ν_size, e_size*ν_size)

        Threads.@threads for x_i in 1:x_size

            e_i, ν_i = x_ind[x_i,:]

            # good credit history
            policy_a_itp = Akima(a_grid, variables.policy_a[:,e_i,ν_i,T_ind])

            if sum(variables.policy_d[:,e_i,ν_i,T_ind]) > 0.0
                a_ind_default = findall(isone, variables.policy_d[:,e_i,ν_i,T_ind])[end]
                a_default = a_grid[a_ind_default]
                a_repay = a_grid[a_ind_default+1]
            else
                a_default = -Inf
                a_repay = a_grid[1]
            end

            for a_i in 1:a_size_μ

                a_μ = a_grid_μ[a_i]

                # repay
                if abs(a_μ - a_repay) <= abs(a_μ - a_default)

                    a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])
                    ind_lower_a_p = findall(a_grid_μ .<= a_p)[end]
                    ind_upper_a_p = findall(a_p .<= a_grid_μ)[1]

                    if ind_lower_a_p != ind_upper_a_p
                        a_lower_a_p = a_grid_μ[ind_lower_a_p]
                        a_upper_a_p = a_grid_μ[ind_upper_a_p]
                        weight_lower = (a_upper_a_p - a_p) / (a_upper_a_p - a_lower_a_p)
                        weight_upper = (a_p - a_lower_a_p) / (a_upper_a_p - a_lower_a_p)
                    else
                        weight_lower = 0.5
                        weight_upper = 0.5
                    end

                    for x_p_i in 1:x_size
                        e_p_i, ν_p_i = x_ind[x_p_i,:]
                        μ_temp[ind_lower_a_p,e_p_i,ν_p_i,x_i] += e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_lower * μ_p[a_i,e_i,ν_i]
                        μ_temp[ind_upper_a_p,e_p_i,ν_p_i,x_i] += e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_upper * μ_p[a_i,e_i,ν_i]
                    end

                # default
                else
                    for x_p_i in 1:x_size
                        e_p_i, ν_p_i = x_ind[x_p_i,:]
                        μ_temp[a_ind_zero_μ,e_p_i,ν_p_i,x_i] += e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * μ_p[a_i,e_i,ν_i]
                    end
                end
            end
        end

        variables.μ[:,:,:,T_ind+1] = dropdims(sum(μ_temp, dims=4), dims=4)
        variables.μ[:,:,:,T_ind+1] = variables.μ[:,:,:,T_ind+1] ./ sum(variables.μ[:,:,:,T_ind+1])
    end
end

function aggregate_func_MIT!(
    variables::mut_var_MIT,
    parameters::NamedTuple
    )

    @unpack x_size, x_ind, e_size, ν_size, a_grid = parameters
    @unpack a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ = parameters
    @unpack ψ_B, ω, r_d = parameters
    @unpack T_size = parameters_MIT

    variables.aggregate_var[:,1:(end-1)] .= 0.0

    for T_ind in 1:(T_size-1)
        # total loans
        for x_i in 1:x_size
            e_i, ν_i = x_ind[x_i,:]
            qa_itp = Akima(a_grid, a_grid.*variables.q[:,e_i,ν_i,(T_ind+1)])
            for a_μ_i in 1:(a_size_neg_μ-1)
                a_μ = a_grid_neg_μ[a_μ_i]
                variables.aggregate_var[1,T_ind] += -(variables.μ[a_μ_i,e_i,ν_i,T_ind+1] * qa_itp(a_μ))
            end
        end

        # total deposits
        variables.aggregate_var[2,T_ind] = sum(variables.μ[(a_ind_zero_μ+1):end,:,:,T_ind+1].*repeat(a_grid_pos_μ[2:end],1,e_size,ν_size))

        # net worth
        if T_ind == 1
            @load "optimal_values.bson" Vss V_repayss V_defautlss qss μss Lss Dss Nss αss λss Λss
            variables.aggregate_var[3,T_ind] = 0.99*Nss
        else
            D = sum(variables.μ[(a_ind_zero_μ+1):end,:,:,T_ind] .* repeat(a_grid_pos_μ[2:end],1,e_size,ν_size))
            L = -sum(variables.μ[1:(a_ind_zero_μ-1),:,:,T_ind] .* repeat(a_grid_neg_μ[1:(end-1)],1,e_size,ν_size))
            L_default = -sum(variables.μ[1:(a_ind_zero_μ-1),:,:,T_ind] .* repeat(a_grid_neg_μ[1:(end-1)],1,e_size,ν_size) .* variables.policy_d_matrix[1:(a_ind_zero_μ-1),:,:,T_ind])
            variables.aggregate_var[3,T_ind] = (1 - ψ_B + ψ_B*ω)*((L-L_default) - (1+r_d)*D)
        end
        # variables.aggregate_var[3,T_ind] = variables.aggregate_var[1,T_ind] - variables.aggregate_var[2,T_ind]

        # asset to debt ratio (or loan to deposit ratio)
        variables.aggregate_var[4,T_ind] = variables.aggregate_var[1,T_ind] / variables.aggregate_var[2,T_ind]

        # leverage ratio
        variables.aggregate_var[5,T_ind] = variables.aggregate_var[1,T_ind] / variables.aggregate_var[3,T_ind]
    end
end

function solve_func_MIT!(
    variables::mut_var_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple;
    tol = 1E-6,
    iter_max = 1E+5
    )

    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving MIT shock: ")

    while crit > tol && iter < iter_max

        if crit > 1E-4
            Δ = 0.4
        else
            Δ = 0.6
        end

        # update the trajectory of leverage ratio
        variables.LN_guess = Δ*variables.aggregate_var[5,:] + (1-Δ)*variables.LN_guess

        # update equilibrium prices
        updating_prics_MIT!(variables, parameters)

        # solve the household's problem (including price schemes)
        # household_func_MIT!(variables, parameters, parameters_MIT)

        value_price_func_MIT!(variables, parameters, parameters_MIT)
        policy_matrix_func_MIT!(variables, parameters)

        # update the cross-sectional distribution
        density_func_MIT!(variables, parameters, parameters_MIT)

        # compute aggregate variables
        aggregate_func_MIT!(variables, parameters)

        # check convergence
        crit = norm(variables.aggregate_var[5,:] .- variables.LN_guess, Inf)

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1

    end

    return ED
end

@load "optimal_values.bson" Vss V_repayss V_defautlss qss μss Lss Dss Nss αss λss Λss
parameters = para_func(; λ = λss)
parameters_MIT = para_func_MIT(parameters; σ_z = 0.00)
variables = var_func_MIT(parameters, parameters_MIT)

# solve_func_MIT!(variables, parameters, parameters_MIT)

New_guess = Δ*variables.aggregate_var[5,:] + (1-Δ)*variables.LN_guess;
plot([variables.LN_guess, variables.aggregate_var[5,:], New_guess],
     label = ["Initial" "Simulated" "Updated"],
     legend = :bottomright,
     lw = 2)

L_per = ((variables.aggregate_var[1,:] .- variables.aggregate_var[1,end]) ./ variables.aggregate_var[1,end])*100
D_per = ((variables.aggregate_var[2,:] .- variables.aggregate_var[2,end]) ./ variables.aggregate_var[2,end])*100
N_per = ((variables.aggregate_var[3,:] .- variables.aggregate_var[3,end]) ./ variables.aggregate_var[3,end])*100
plot([L_per, D_per, N_per],
     label = ["Loans" "Deposits" "Net Worth"],
     lw = 2)

N_diff = variables.aggregate_var[1,:] .- variables.aggregate_var[2,:]
plot([variables.aggregate_var[3,:], N_diff],
     label = ["true" "difference"],
     legend = :bottomright,
     lw = 2)
