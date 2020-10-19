include("FLOWMath.jl")
using Main.FLOWMath: Akima, akima, interp2d
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
    # λ = 1 - β_H*ψ_B*(1+r_d) - 10^(-4)
    # λ = 1.0 - (β_B*ψ_B*(1+r_d))^(1/2)
    α = (β_B*(1.0-ψ_B)*(1.0+r_d)) / ((1.0-λ)-β_B*ψ_B*(1.0+r_d))
    Λ = β_B*(1.0-ψ_B+ψ_B*α)
    r_ld = λ*θ/Λ
    LR = α/θ
    AD = LR/(LR-1)
    # ω = ((1.0-ψ_B)^(-1.0)) * (((1.0+r_d+r_ld)*LR-(1.0+r_d))^(-1.0) - ψ_B)
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

mutable struct mut_var
    q::Array{Float64,3}
    prob_default::Array{Float64,3}
    V::Array{Float64,3}
    V_repay::Array{Float64,3}
    V_default::Array{Float64,3}
    policy_a::Array{Float64,3}
    policy_a_matrix::Array{Float64,4}
    policy_d::Array{Float64,3}
    policy_d_matrix::Array{Float64,3}
    μ::Array{Float64,3}
    μ_Γ::SparseMatrixCSC{Float64,Int64}
    aggregate_var::Array{Float64,1}
end

function var_func(
    parameters::NamedTuple;
    load_initial_values::Integer = 0
    )
    #------------------------------------------------------------#
    # construct a mutable object containing endogenous variables #
    #------------------------------------------------------------#

    # unpack parameters
    @unpack a_grid, a_size, a_size_pos, a_ind_zero = parameters
    @unpack a_size_μ = parameters
    @unpack x_size, ν_size, e_size, e_grid = parameters
    @unpack σ, r_d, r_ld = parameters

    if load_initial_values == 1
        @load "24092020_initial_values.bson" V q μ

        # define default probability
        prob_default = zeros(a_size, e_size, ν_size)

        # define value functions
        V_repay = zeros(a_size, e_size, ν_size)
        V_default = zeros(a_size, e_size, ν_size)

        # define policy functions
        policy_a = zeros(a_size, e_size, ν_size)
        policy_a_matrix = zeros(a_size_μ, a_size_μ, e_size, ν_size)
        policy_d = zeros(a_size, e_size, ν_size)
        policy_d_matrix = zeros(a_size_μ, e_size, ν_size)

        # define the type distribution and its transition matrix
        μ_size = x_size*a_size_μ
        μ_Γ = spzeros(μ_size, μ_size)
    else
        # define pricing related variables
        q = ones(a_size, e_size, ν_size)
        q[findall(a_grid .< 0.0),:,:] .= 1.0 / (1.0 + r_d + r_ld)

        # define default probability
        prob_default = zeros(a_size, e_size, ν_size)

        # define value functions
        V = zeros(a_size, e_size, ν_size)
        V_repay = zeros(a_size, e_size, ν_size)
        V_default = zeros(a_size, e_size, ν_size)

        # define policy functions
        policy_a = zeros(a_size, e_size, ν_size)
        policy_a_matrix = zeros(a_size_μ, a_size_μ, e_size, ν_size)
        policy_d = zeros(a_size, e_size, ν_size)
        policy_d_matrix = zeros(a_size_μ, e_size, ν_size)

        # define the type distribution and its transition matrix
        μ_size = x_size*a_size_μ
        μ = ones(a_size_μ, e_size, ν_size) ./ μ_size
        μ_Γ = spzeros(μ_size, μ_size)
    end

    # define aggregate objects
    aggregate_var = zeros(4)

    # return outputs
    variables = mut_var(q, prob_default, V, V_repay, V_default,
                        policy_a, policy_a_matrix, policy_d, policy_d_matrix,
                        μ, μ_Γ, aggregate_var)
    return variables
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

function value_func!(
    V_p::Array{Float64,3},
    q_p::Array{Float64,3},
    variables::mut_var,
    parameters::NamedTuple
    )

    @unpack a_grid, a_grid_pos, a_ind_zero = parameters
    @unpack a_size, a_size_pos = parameters
    @unpack x_grid, x_size, x_ind, x_Γ = parameters
    @unpack e_grid, e_Γ, ν_p = parameters
    @unpack β_H, σ, ξ, r_d, r_ld = parameters

    Threads.@threads for x_i in 1:x_size

        e, ν = x_grid[x_i,:]
        e_i, ν_i = x_ind[x_i,:]

        V_expt_p = (ν_p*V_p[:,:,1] + (1-ν_p)*V_p[:,:,2])*e_Γ[e_i,:]
        V_hat = (ν*β_H)*V_expt_p
        V_hat_itp = Akima(a_grid, V_hat)

        variables.V_default[:,e_i,ν_i] .= u_func((1-ξ)*exp(e), σ) + V_hat[a_ind_zero]

        q = q_p[:,e_i,ν_i]
        qa = q .* a_grid
        qa_itp = Akima(a_grid, qa)

        for a_i in 1:a_size
            a = a_grid[a_i]
            CoH = exp(e) + (1+r_d*(a>0))*a

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
            variables.V_repay[a_i,e_i,ν_i] = -Optim.minimum(res_good)
            if variables.V_default[a_i,e_i,ν_i] > variables.V_repay[a_i,e_i,ν_i]
                variables.V[a_i,e_i,ν_i] = variables.V_default[a_i,e_i,ν_i]
                variables.policy_a[a_i,e_i,ν_i] = 0.0
                variables.policy_d[a_i,e_i,ν_i] = 1.0
            else
                variables.V[a_i,e_i,ν_i] = variables.V_repay[a_i,e_i,ν_i]
                variables.policy_a[a_i,e_i,ν_i] = Optim.minimizer(res_good)[]
                variables.policy_d[a_i,e_i,ν_i] = 0.0
            end
        end
    end
end

function price_func!(
    q_p::Array{Float64,3},
    variables::mut_var,
    parameters::NamedTuple
    )
    #----------------------------#
    # update the price schedule. #
    #----------------------------#
    @unpack ξ, r_d, r_ld = parameters
    @unpack a_size, a_grid, a_grid_neg, a_size_neg = parameters
    @unpack x_ind, x_size, x_grid, x_Γ = parameters
    @unpack e_size, e_grid, e_ρ, e_σ = parameters
    @unpack ν_p, ν_Γ, ν_size = parameters

    Δ = 0.7    # parameter controling update speed

    a_grid_neg_nozero = a_grid_neg[1:(end-1)]
    a_size_neg_nozero = length(a_grid_neg_nozero)
    q_update = ones(a_size_neg_nozero, e_size, ν_size)

    Threads.@threads for a_p_i in 1:a_size_neg_nozero

        # impatient household
        V_diff_1 = variables.V_repay[a_p_i,:,1] .- variables.V_default[a_p_i,:,1]
        if all(V_diff_1 .> 0.0)
            e_p_thres_1 = -Inf
        elseif all(V_diff_1 .< 0.0)
            e_p_thres_1 = Inf
        else
            # e_p_lower_1 = e_grid[searchsortedlast(V_diff_1, 0.0)]
            # e_p_upper_1 = e_grid[searchsortedfirst(V_diff_1, 0.0)]
            e_p_lower_1 = e_grid[findall(V_diff_1 .<= 0.0)[end]]
            e_p_upper_1 = e_grid[findall(V_diff_1 .>= 0.0)[1]]
            V_diff_1_itp = Akima(e_grid, V_diff_1)
            object_V_diff_1(e_p) = V_diff_1_itp(e_p)
            e_p_thres_1 = find_zero(object_V_diff_1, (e_p_lower_1, e_p_upper_1), Bisection())
        end

        # patient household
        V_diff_2 = variables.V_repay[a_p_i,:,2] .- variables.V_default[a_p_i,:,2]
        if all(V_diff_2 .> 0.0)
            e_p_thres_2 = -Inf
        elseif all(V_diff_2 .< 0.0)
            e_p_thres_2 = Inf
        else
            # e_p_lower_2 = e_grid[searchsortedlast(V_diff_2, 0.0)]
            # e_p_upper_2 = e_grid[searchsortedfirst(V_diff_2, 0.0)]
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
            repay_prob = ν_p*(1.0-default_prob_1) + (1.0-ν_p)*(1.0-default_prob_2)

            # garnishment_1 = cdf(dist, (e_p_thres_1-(e_ρ*e+e_σ^2))/e_σ)
            # garnishment_2 = cdf(dist, (e_p_thres_2-(e_ρ*e+e_σ^2))/e_σ)
            # garnishment_rate = (ξ/-a_grid[a_p_i])*exp(e_ρ*e + (e_σ^2/2.0))*(ν_p*garnishment_1 + (1.0-ν_p)*garnishment_2)

            q_update[a_p_i,e_i,:] .= repay_prob / (1.0+r_d+r_ld)
            variables.prob_default[a_p_i,e_i,:] .= 1.0 - repay_prob
            # q_update[a_p_i,e_i,:] .= (repay_prob+garnishment_rate) / (1.0+r_d+r_ld)
        end

    end
    variables.q[1:a_size_neg_nozero,:,:] = Δ*q_update + (1-Δ)*q_p[1:a_size_neg_nozero,:,:]
end

function household_func!(
    variables::mut_var,
    parameters::NamedTuple;
    tol = tol_h,
    iter_max = iter_max
    )

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving household's maximization (one loop): ")

    # initialize the next-period value functions
    V_p = similar(variables.V)
    q_p = similar(variables.q)

    while crit > tol && iter < iter_max

        # println("iter = $iter")
        # copy the current value functions to the pre-specified containers
        copyto!(V_p, variables.V)
        copyto!(q_p, variables.q)

        # update value function
        value_func!(V_p, q_p, variables, parameters)

        # update price, its derivative, and size of bond
        price_func!(q_p, variables, parameters)

        # check convergence
        crit = max(norm(variables.V .- V_p, Inf), norm(variables.q .- q_p, Inf))

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end

function policy_matrix_func!(
    variables::mut_var,
    parameters::NamedTuple
    )

    @unpack x_size, x_ind, a_grid, a_size = parameters
    @unpack a_size_μ, a_grid_μ, a_ind_zero_μ = parameters

    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        policy_a_itp = Akima(a_grid, variables.policy_a[:,e_i,ν_i])

        if sum(variables.policy_d[:,e_i,ν_i]) > 0.0
            a_ind_default = findall(isone,variables.policy_d[:,e_i,ν_i])[end]
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
                # ind_lower_a_p = searchsortedlast(a_grid_μ, a_p)
                # ind_upper_a_p = searchsortedfirst(a_grid_μ, a_p)
                ind_lower_a_p = findall(a_grid_μ .<= a_p)[end]
                ind_upper_a_p = findall(a_p .<= a_grid_μ)[1]
                if ind_lower_a_p != ind_upper_a_p
                    a_lower_a_p = a_grid_μ[ind_lower_a_p]
                    a_upper_a_p = a_grid_μ[ind_upper_a_p]
                    weight_lower = (a_upper_a_p - a_p) / (a_upper_a_p - a_lower_a_p)
                    weight_upper = (a_p - a_lower_a_p) / (a_upper_a_p - a_lower_a_p)
                    variables.policy_a_matrix[a_i,ind_lower_a_p,e_i,ν_i] = weight_lower
                    variables.policy_a_matrix[a_i,ind_upper_a_p,e_i,ν_i] = weight_upper
                else
                    variables.policy_a_matrix[a_i,ind_lower_a_p,e_i,ν_i] = 1.0
                end
                variables.policy_d_matrix[a_i,e_i,ν_i] = 0.0
            # default
            else
                variables.policy_a_matrix[a_i,a_ind_zero_μ,e_i,ν_i] = 1.0
                variables.policy_d_matrix[a_i,e_i,ν_i] = 1.0
            end
        end
    end
end

function density_func!(
    variables::mut_var,
    parameters::NamedTuple;
    tol = tol_μ,
    iter_max = iter_max
    )

    @unpack x_size, x_ind, e_Γ, ν_Γ, e_size, ν_size = parameters
    @unpack a_size, a_grid, a_size_pos, a_grid_pos = parameters
    @unpack a_size_μ, a_grid_μ, a_size_pos_μ, a_grid_pos_μ, a_ind_zero_μ = parameters

    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving invariant density: ")

    μ_p = similar(variables.μ)

    while crit > tol && iter < iter_max

        copyto!(μ_p, variables.μ)
        variables.μ .= 0.0
        μ_temp = zeros(a_size_μ, e_size, ν_size, e_size*ν_size)

        Threads.@threads for x_i in 1:x_size

            e_i, ν_i = x_ind[x_i,:]

            # good credit history
            policy_a_itp = Akima(a_grid, variables.policy_a[:,e_i,ν_i])

            if sum(variables.policy_d[:,e_i,ν_i]) > 0.0
                a_ind_default = findall(isone, variables.policy_d[:,e_i,ν_i])[end]
                a_default = a_grid[a_ind_default]
                a_repay = a_grid[a_ind_default+1]
            else
                a_default = -Inf
                a_repay = a_grid[1]
            end

            for a_i in 1:a_size_μ

                a_μ = a_grid_μ[a_i]

                # good -> good
                if abs(a_μ - a_repay) <= abs(a_μ - a_default)

                    a_p = clamp(policy_a_itp(a_grid_μ[a_i]), a_grid[1], a_grid[end])
                    # ind_lower_a_p = searchsortedlast(a_grid_μ, a_p)
                    # ind_upper_a_p = searchsortedfirst(a_grid_μ, a_p)
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

                # good -> bad
                else
                    for x_p_i in 1:x_size
                        e_p_i, ν_p_i = x_ind[x_p_i,:]
                        μ_temp[a_ind_zero_μ,e_p_i,ν_p_i,x_i] += e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * μ_p[a_i,e_i,ν_i]
                    end
                end
            end
        end
        variables.μ .= dropdims(sum(μ_temp, dims=4), dims=4)
        variables.μ .= variables.μ ./ sum(variables.μ)

        # check convergence
        crit = norm(variables.μ .- μ_p, Inf)

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end

function aggregate_func!(
    variables::mut_var,
    parameters::NamedTuple
    )

    @unpack x_size, x_ind, e_size, ν_size, a_grid = parameters
    @unpack a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ = parameters

    # total loans
    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        qa_itp = Akima(a_grid, a_grid.*variables.q[:,e_i,ν_i])
        for a_μ_i in 1:(a_size_neg_μ-1)
            a_μ = a_grid_neg_μ[a_μ_i]
            variables.aggregate_var[1] += -(variables.μ[a_μ_i,e_i,ν_i] * qa_itp(a_μ))
        end
    end

    # total deposits
    variables.aggregate_var[2] = sum(variables.μ[(a_ind_zero_μ+1):end,:,:].*repeat(a_grid_pos_μ[2:end],1,e_size,ν_size))

    # net worth
    variables.aggregate_var[3] = variables.aggregate_var[1] - variables.aggregate_var[2]

    # asset to debt ratio (or loan to deposit ratio)
    variables.aggregate_var[4] = variables.aggregate_var[1] / variables.aggregate_var[2]
end

function solve_func!(
    variables::mut_var,
    parameters::NamedTuple;
    tol_h = 1E-8,
    tol_μ = 1E-10,
    iter_max = 500
    )

    # solve the household's problem (including price schemes)
    household_func!(variables, parameters; tol = tol_h, iter_max = iter_max)

    # represent policy functions in matrices
    policy_matrix_func!(variables, parameters)

    # update the cross-sectional distribution
    density_func!(variables, parameters; tol = tol_μ, iter_max = iter_max)

    # compute aggregate variables
    aggregate_func!(variables, parameters)

    ED = variables.aggregate_var[4] - parameters.AD

    data_spec = Any[#=1=# "Multiplier"                   parameters.λ;
                    #=2=# "Asset-to-Debt Ratio (Demand)" variables.aggregate_var[4];
                    #=3=# "Asset-to-Debt Ratio (Supply)" parameters.AD;
                    #=4=# "Difference"                   ED]

    pretty_table(data_spec, ["Name", "Value"];
                 alignment=[:l,:r],
                 formatters = ft_round(12),
                 body_hlines = [1,3])

    # save results
    V = variables.V
    q = variables.q
    μ = variables.μ
    @save "24092020_initial_values.bson" V q μ

    return ED
end

#=
parameters = para_func(;λ = 0.06)
variables = var_func(parameters)

println("Solving the model with $(Threads.nthreads()) threads in Julia...")
solve_func!(variables, parameters)

data_spec = Any[#= 1=# "Number of Endowment"                parameters.e_size;
                #= 2=# "Number of Assets"                   parameters.a_size;
                #= 3=# "Number of Negative Assets"          parameters.a_size_neg;
                #= 4=# "Number of Positive Assets"          parameters.a_size_pos;
                #= 5=# "Number of Assets (for Density)"     parameters.a_size_μ;
                #= 6=# "Minimum of Assets"                  parameters.a_grid[1];
                #= 7=# "Maximum of Assets"                  parameters.a_grid[end];
                #= 8=# "Scale of Impatience"                parameters.ν_grid[1];
                #= 9=# "Probability of being Impatient"     parameters.ν_Γ[1,1];
                #=10=# "Exogenous Risk-free Rate"           parameters.r_d;
                #=11=# "Multiplier of Incentive Constraint" parameters.λ;
                #=12=# "Marginal Benifit of Net Worth"      parameters.α;
                #=13=# "Diverting Fraction"                 parameters.θ;
                #=14=# "Asset-to-Debt Ratio (Supply)"       parameters.AD;
                #=15=# "Additional Opportunity Cost"        parameters.r_ld;
                #=16=# "Total Loans"                        variables.aggregate_var[1];
                #=17=# "Total Deposits"                     variables.aggregate_var[2];
                #=18=# "Net Worth"                          variables.aggregate_var[3];
                #=19=# "Asset-to-Debt Ratio (Demand)"       variables.aggregate_var[4]]

hl_LR = Highlighter(f      = (data,i,j) -> i == 14 || i == 19,
                    crayon = Crayon(background = :light_blue))

pretty_table(data_spec, ["Name", "Value"];
             alignment=[:l,:r],
             formatters = ft_round(4),
             body_hlines = [7,9,15],
             highlighters = hl_LR)
=#

#=
parameters = para_func()
para_targeted(x) = para_func(; λ = x)
solve_targeted(x) = solve_func!(var_func(para_targeted(x)), para_targeted(x))
λ_lower, λ_upper = 0, 1-(parameters.β_B*parameters.ψ_B*(1+parameters.r_d))^(1/2)
# λ_optimal = find_zero(solve_targeted, (λ_lower, λ_upper), Bisection())
λ_optimal = find_zero(solve_targeted, (0.04987767453057565, 0.05), Bisection())
=#

#=
λ_optimal = 0.04988090645870582
parameters = para_func(; λ = λ_optimal)
variables = var_func(parameters)
println("Solving the model with $(Threads.nthreads()) threads in Julia...")
solve_func!(variables, parameters)

Vss = variables.V
V_repayss = variables.V_repay
V_defautlss = variables.V_default
qss = variables.q
μss = variables.μ
Lss = variables.aggregate_var[1]
Dss = variables.aggregate_var[2]
Nss = variables.aggregate_var[3]
λss = parameters.λ
αss = parameters.α
Λss = parameters.Λ

@save "optimal_values.bson" Vss V_repayss V_defautlss qss μss Lss Dss Nss αss λss Λss
=#

#=========================================================#
# Comparison between with and without financial frictions #
#=========================================================#
#=
λ_optimal = 0.04988090645870582
parameters_FF = para_func(; λ = λ_optimal, e_σ = 0.10)
variables_FF = var_func(parameters_FF)
solve_func!(variables_FF, parameters_FF)

parameters_NFF = para_func(; λ = 0.0, θ = 0.0, e_σ = 0.10)
variables_NFF = var_func(parameters_NFF)
solve_func!(variables_NFF, parameters_NFF)

results_compare_FF = zeros(6,3)
results_compare_FF[1:4,1] = variables_NFF.aggregate_var
results_compare_FF[5,1] = 0.0
results_compare_FF[6,1] = sum(variables_NFF.μ.*variables_NFF.policy_d_matrix)*100
results_compare_FF[1:3,2] = variables_FF.aggregate_var[1:3]
results_compare_FF[4,2] = variables_FF.aggregate_var[1]/variables_FF.aggregate_var[3]
results_compare_FF[5,2] = parameters_FF.r_ld
results_compare_FF[6,2] = sum(variables_FF.μ.*variables_FF.policy_d_matrix)*100
results_compare_FF[:,3] = ((results_compare_FF[:,2] .- results_compare_FF[:,1])./results_compare_FF[:,1])*100
=#
#=
PD_FF = parameters_FF.ν_p*variables_FF.prob_default[1:parameters_FF.a_ind_zero,:,1] + (1-parameters_FF.ν_p)*variables_FF.prob_default[1:parameters_FF.a_ind_zero,:,2]
PD_NFF = parameters_NFF.ν_p*variables_NFF.prob_default[1:parameters_NFF.a_ind_zero,:,1] + (1-parameters_NFF.ν_p)*variables_NFF.prob_default[1:parameters_NFF.a_ind_zero,:,2]

step = 10^(-2)
e_itp_FF = parameters_FF.e_grid[1]:step:parameters_FF.e_grid[end]
a_p_itp_FF = parameters_FF.a_grid_neg[1]:step:parameters_FF.a_grid_neg[end]
PD_FF_itp = interp2d(akima, parameters_FF.a_grid_neg, parameters_FF.e_grid, PD_FF, a_p_itp_FF, e_itp_FF)

plot_PD_FF = plot(e_itp_FF, a_p_itp_FF, PD_FF_itp,
                  st = :heatmap, clim = (0,1), color = :dense,
                  xlabel = "\$p\$", ylabel = "\$a'\$",
                  colorbar_title = "\$P(d'=1)\$")

e_itp_NFF = parameters_NFF.e_grid[1]:step:parameters_NFF.e_grid[end]
a_p_itp_NFF = parameters_NFF.a_grid_neg[1]:step:parameters_NFF.a_grid_neg[end]
PD_NFF_itp = interp2d(akima, parameters_NFF.a_grid_neg, parameters_NFF.e_grid, PD_NFF, a_p_itp_NFF, e_itp_NFF)

plot_PD_NFF = plot(e_itp_NFF, a_p_itp_NFF, PD_NFF_itp,
                   st = :heatmap, clim = (0,1), color = :dense,
                   xlabel = "\$p\$", ylabel = "\$a'\$",
                   colorbar_title = "\$P(d'=1)\$")

plot(plot_PD_FF, plot_PD_NFF, layout = (2,1))
savefig("plot_PD_FF.pdf")
=#

#========================================#
# Comparison of time-varying uncertainty #
#========================================#
#=
parameters = para_func()
para_targeted(x) = para_func(; λ = x, e_σ = 0.10*1.02)
solve_targeted(x) = solve_func!(var_func(para_targeted(x)), para_targeted(x))
λ_lower, λ_upper = 0, 1-(parameters.β_B*parameters.ψ_B*(1+parameters.r_d))^(1/2)
# λ_optimal_u = find_zero(solve_targeted, (λ_lower, λ_upper), Bisection())
λ_optimal_u = find_zero(solve_targeted, (0.026556104511, 0.026556104541), Bisection())
=#

#=
λ_optimal_u = 0.026556104511
parameters_FF_u = para_func(; λ = λ_optimal_u, e_σ = 0.10*1.02)
variables_FF_u = var_func(parameters_FF_u)
solve_func!(variables_FF_u, parameters_FF_u)

results_compare_u = zeros(6,3)
results_compare_u[1:3,1] = variables_FF.aggregate_var[1:3]
results_compare_u[4,1] = variables_FF.aggregate_var[1] / variables_FF.aggregate_var[3]
results_compare_u[5,1] = parameters_FF.r_ld
results_compare_u[6,1] = sum(variables_FF.μ.*variables_FF.policy_d_matrix)*100

results_compare_u[1:3,2] = variables_FF_u.aggregate_var[1:3]
results_compare_u[4,2] = variables_FF_u.aggregate_var[1]/variables_FF_u.aggregate_var[3]
results_compare_u[5,2] = parameters_FF_u.r_ld
results_compare_u[6,2] = sum(variables_FF_u.μ.*variables_FF_u.policy_d_matrix)*100

results_compare_u[:,3] = ((results_compare_u[:,2] .- results_compare_u[:,1])./results_compare_u[:,1])*100
=#

#=
#=======================================#
# Comparison of time-varying preference #
#=======================================#
parameters = para_func()
para_targeted(x) = para_func(; λ = x, ν_p = 0.10*1.02)
solve_targeted(x) = solve_func!(var_func(para_targeted(x)), para_targeted(x))
λ_lower, λ_upper = 0, 1-(parameters.β_B*parameters.ψ_B*(1+parameters.r_d))^(1/2)
# λ_optimal_u = find_zero(solve_targeted, (λ_lower, λ_upper), Bisection())
λ_optimal_ν = find_zero(solve_targeted, (0.04988090645870582, 0.06), Bisection())

λ_optimal_ν = 0.05916473242601564
parameters_FF_ν = para_func(; λ = λ_optimal_ν, ν_p = 0.10*1.02)
variables_FF_ν = var_func(parameters_FF_ν)
solve_func!(variables_FF_ν, parameters_FF_ν)

results_compare_ν = zeros(6,3)
results_compare_ν[1:3,1] = variables_FF.aggregate_var[1:3]
results_compare_ν[4,1] = variables_FF.aggregate_var[1] / variables_FF.aggregate_var[3]
results_compare_ν[5,1] = parameters_FF.r_ld
results_compare_ν[6,1] = sum(variables_FF.μ.*variables_FF.policy_d_matrix)*100

results_compare_ν[1:3,2] = variables_FF_ν.aggregate_var[1:3]
results_compare_ν[4,2] = variables_FF_ν.aggregate_var[1]/variables_FF_ν.aggregate_var[3]
results_compare_ν[5,2] = parameters_FF_ν.r_ld
results_compare_ν[6,2] = sum(variables_FF_ν.μ.*variables_FF_ν.policy_d_matrix)*100

results_compare_ν[:,3] = ((results_compare_ν[:,2] .- results_compare_ν[:,1])./results_compare_ν[:,1])*100
=#

#=
parameters_NFF_u = para_func(; λ = 0.0, θ = 0.0, e_σ = 0.10*1.02)
variables_NFF_u = var_func(parameters_NFF_u)
solve_func!(variables_NFF_u, parameters_NFF_u)
=#

#=
plot(parameters.a_grid_neg, variables.q[1:parameters.a_ind_zero,:,1], seriestype=:scatter, legend=:bottomright)
plot(parameters.a_grid_neg, repeat(parameters.a_grid_neg,1,parameters.e_size).*variables.q[1:parameters.a_ind_zero,:,1], seriestype=:scatter, legend=:bottomright)

function plot_price_itp(
    a_size_neg_itp::Integer,
    variables::mut_var,
    parameters::NamedTuple)

    @unpack e_size, a_grid, a_grid = parameters

    q_all = zeros(a_size_neg_itp, e_size)
    a_grid_itp = collect(range(a_grid[1], 0, length = a_size_neg_itp))

    for e_i in 1:e_size
        q_itp = interpolate(a_grid, variables.q[:,e_i])
        q_all[:, e_i] .= [q_itp(a_grid_itp[a_i]) for a_i in 1:a_size_neg_itp]
    end

    return q_all, a_grid_itp
end
q_all, a_grid_itp = plot_price_itp(5000, variables, parameters)
plot(a_grid_itp, q_all)
plot(a_grid_itp, q_all .* repeat(a_grid_itp,1,parameters.e_size))
=#

#=
β_B = 0.90
r_d = 0.02
ψ_B = 0.8
θ = 0.20
L_λ = 0
U_λ = 1 - β_B*ψ_B*(1+r_d)
Δ_λ = 10^(-2)
λ_grid = collect(L_λ:Δ_λ:U_λ)
α(λ) = (β_B*(1-ψ_B)*(1+r_d)) / ((1-λ)-β_B*ψ_B*(1+r_d))
Λ(λ) = β_B*(1-ψ_B+ψ_B*α(λ))
C(λ) = λ*θ / Λ(λ)
λ_optimal = 1-√(β_B*ψ_B*(1+r_d))
L_α = α(L_λ)
L_Λ = Λ(L_λ)
U_α = α(λ_optimal)
U_Λ = Λ(λ_optimal)
L_C = 0
U_C = C(λ_optimal)
α_grid = α.(λ_grid)
Λ_grid = Λ.(λ_grid)
C_grid = C.(λ_grid)

table = round.([L_λ λ_optimal; L_α U_α; L_Λ U_Λ; L_C U_C; L_α/θ U_α/θ],digits=4)


plot(λ_grid, [α_grid Λ_grid], seriestype=:scatter)
plot(λ_grid, C_grid, lw = 2, label = "C(λ)")
scatter!([λ_optimal], [U_C], label = "optimal λ")
savefig("plot_C_lambda.pdf")
=#

#=
PD = 1 .- variables.q[1:parameters.a_size_neg,1:parameters.p_size,1] .* (1+parameters.r_f+parameters.r_bf)
plot(parameters.e_grid, parameters.a_grid_neg, variables.prob_default[1:parameters.a_ind_zero,:,1],
     st = :heatmap, clim = (0,1), color = :dense,
     xlabel = "\$e\$", ylabel = "\$a'\$",
     xlims = (-1,0), ylims = (-1.0,0.0),
     yticks = parameters.a_grid_neg[1]:0.5:parameters.a_grid_neg[end],
     colorbar_title = "\$P(g'_h(a',s)=1)\$",
     theme = theme(:ggplot2))
=#
