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

function para_func(;
    ψ_H::Real = 0.10,           # history rased probability
    ψ_B::Real = 0.80,           # bank's survival rate
    β_H::Real = 0.96,           # discount factor (households)
    β_B::Real = 0.90,           # discount factor (banks)
    ξ::Real = 0.35,             # garnishment rate
    σ::Real = 2.0,              # CRRA coefficient
    z::Real = 1.0,              # aggregate uncertainty
    r_d::Real = 0.03,           # deposit rate
    λ::Real = 0.00,             # the multiplier of incentive constraint
    θ::Real = 0.20,             # the diverted fraction
    e_ρ::Real = 0.95,           # AR(1) of earnings shock
    e_σ::Real = 0.10,           # s.d. of earni ngs shock
    e_size::Integer = 15,       # no. of earnings shock
    ν_size::Integer = 2,        # no. of preference shock
    ν_s::Real = 0.80,           # scale of patience
    ν_p::Real = 0.05,           # probability of patience
    a_min::Real = -4.5,         # min of asset holding
    a_max::Real = 350.0,        # max of asset holding
    a_size_neg::Integer = 1351, # number of the grid of negative asset holding for VFI
    a_size_pos::Integer = 351,  # number of the grid of positive asset holding for VFI
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
    a_ind_zero = findall(a_grid .== 0.0)[1]

    # asset holding grid for μ
    a_size_neg_μ = convert(Int, a_size_neg*μ_scale)
    a_grid_neg_μ = collect(range(a_min, 0.0, length = a_size_neg_μ))
    a_size_pos_μ = convert(Int, a_size_pos*μ_scale)
    a_grid_pos_μ = collect(range(0.0, a_max, length = a_size_pos_μ))
    a_grid_μ = cat(a_grid_neg_μ, a_grid_pos_μ[2:end], dims = 1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(a_grid_μ .== 0.0)[1]

    # auxiliary indicies for density
    μ_good_ind = gridmake(1:a_size_μ, 1:e_size, 1:ν_size)
    μ_bad_ind = gridmake(1:a_size_pos_μ, 1:e_size, 1:ν_size)

    # solve the steady state of ω and θ to match targeted parameters
    # λ = 1 - β_H*ψ_B*(1+r_d) - 10^(-4)
    λ = 1.0 - (β_B*ψ_B*(1+r_d))^(1/2)
    α = (β_B*(1.0-ψ_B)*(1.0+r_d)) / ((1.0-λ)-β_B*ψ_B*(1.0+r_d))
    Λ = β_B*(1.0-ψ_B+ψ_B*α)
    r_ld = λ*θ/Λ
    LR = α/θ
    ω = ((1.0-ψ_B)^(-1.0)) * (((1.0+r_d+r_ld)*LR-(1.0+r_d))^(-1.0) - ψ_B)

    # return values
    return (ψ_H = ψ_H, ψ_B = ψ_B, β_H = β_H, β_B = β_B, ξ = ξ, σ = σ, z = z,
            LR = LR, r_ld = r_ld, r_d = r_d, θ = θ, λ = λ, α = α, Λ = Λ, ω = ω,
            a_grid = a_grid, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos,
            a_size = a_size, a_size_neg = a_size_neg, a_size_pos = a_size_pos,
            a_ind_zero = a_ind_zero,
            a_grid_μ = a_grid_μ, a_grid_neg_μ = a_grid_neg_μ, a_grid_pos_μ = a_grid_pos_μ,
            a_size_μ = a_size_μ, a_size_neg_μ = a_size_neg_μ, a_size_pos_μ = a_size_pos_μ,
            a_ind_zero_μ = a_ind_zero_μ, μ_scale = μ_scale,
            μ_good_ind = μ_good_ind, μ_bad_ind = μ_bad_ind,
            e_ρ = e_ρ, e_σ = e_σ, e_Γ = e_Γ, e_grid = e_grid, e_size = e_size,
            ν_p = ν_p, ν_Γ = ν_Γ, ν_grid = ν_grid, ν_size = ν_size,
            x_Γ = x_Γ, x_grid = x_grid, x_size = x_size, x_ind = x_ind)
end

mutable struct mut_var
    q::Array{Float64,3}
    prob_default::Array{Float64,3}
    V_good::Array{Float64,3}
    V_good_repay::Array{Float64,3}
    V_good_default::Array{Float64,3}
    V_bad::Array{Float64,3}
    policy_a_good::Array{Float64,3}
    policy_a_bad::Array{Float64,3}
    policy_d_good::Array{Float64,3}
    μ_good::Array{Float64,3}
    μ_bad::Array{Float64,3}
    μ_Γ::SparseMatrixCSC{Float64,Int64}
    aggregate_var::Array{Float64,1}
end

function var_func(
    parameters::NamedTuple;
    lond_initial_values::Integer = 1
    )
    #------------------------------------------------------------#
    # construct a mutable object containing endogenous variables #
    #------------------------------------------------------------#

    # unpack parameters
    @unpack a_grid, a_size, a_size_pos, a_ind_zero = parameters
    @unpack a_size_μ, a_size_pos_μ = parameters
    @unpack x_size, ν_size, e_size, e_grid = parameters
    @unpack σ, r_d, r_ld = parameters

    if lond_initial_values == 1
        @load "24092020_initial_values.bson" V_good V_bad q μ_good μ_bad

        # define default probability
        prob_default = zeros(a_size, e_size, ν_size)

        # define value functions
        V_good_repay = zeros(a_size, e_size, ν_size)
        V_good_default = zeros(a_size, e_size, ν_size)

        # define policy functions
        policy_a_good = zeros(a_size, e_size, ν_size)
        policy_a_bad = zeros(a_size_pos, e_size, ν_size)
        policy_d_good = zeros(a_size, e_size, ν_size)

        # define the type distribution and its transition matrix
        μ_size = x_size*(a_size_μ + a_size_pos_μ)
        μ_Γ = spzeros(μ_size, μ_size)
    else
        # define pricing related variables
        q = ones(a_size, e_size, ν_size)
        q[findall(a_grid .< 0.0),:,:] .= 1.0 / (1.0 + r_d + r_ld)

        # define default probability
        prob_default = zeros(a_size, e_size, ν_size)

        # define value functions
        V_good = zeros(a_size, e_size, ν_size)
        V_good_repay = zeros(a_size, e_size, ν_size)
        V_good_default = zeros(a_size, e_size, ν_size)
        V_bad = zeros(a_size_pos, e_size, ν_size)

        # V_good = u_func.(repeat(e_grid',a_size,ν_size) .+ repeat((1.0 .+ r_d*(a_grid.>0.0)).*a_grid,1,x_size), σ)
        # copyto!(V_bad, V_good[a_ind_zero:end,:])

        # define policy functions
        policy_a_good = zeros(a_size, e_size, ν_size)
        policy_a_bad = zeros(a_size_pos, e_size, ν_size)
        policy_d_good = zeros(a_size, e_size, ν_size)

        # define the type distribution and its transition matrix
        μ_size = x_size*(a_size_μ + a_size_pos_μ)
        μ_good = ones(a_size_μ, e_size, ν_size) ./ μ_size
        μ_bad = ones(a_size_pos_μ, e_size, ν_size) ./ μ_size
        μ_Γ = spzeros(μ_size, μ_size)
    end

    # define aggregate objects
    aggregate_var = zeros(4)

    # return outputs
    variables = mut_var(q, prob_default, V_good, V_good_repay, V_good_default, V_bad, policy_a_good, policy_a_bad, policy_d_good, μ_good, μ_bad, μ_Γ, aggregate_var)
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
    V_good_p::Array{Float64,3},
    V_bad_p::Array{Float64,3},
    q_p::Array{Float64,3},
    variables::mut_var,
    parameters::NamedTuple
    )

    @unpack a_grid, a_grid_pos, a_ind_zero = parameters
    @unpack a_size, a_size_pos = parameters
    @unpack x_grid, x_size, x_ind, x_Γ = parameters
    @unpack e_grid, e_Γ, ν_p = parameters
    @unpack β_H, σ, ξ, ψ_H, r_d, r_ld = parameters

    Threads.@threads for x_i in 1:x_size

        e, ν = x_grid[x_i,:]
        e_i, ν_i = x_ind[x_i,:]

        # bad credit history
        V_bad_g_p = (ν_p*V_good_p[a_ind_zero:end,:,1] + (1-ν_p)*V_good_p[a_ind_zero:end,:,2])*e_Γ[e_i,:]
        V_bad_b_p = (ν_p*V_bad_p[:,:,1] + (1-ν_p)*V_bad_p[:,:,2])*e_Γ[e_i,:]
        V_hat_bad = (ν*β_H)*(ψ_H*V_bad_g_p .+ (1-ψ_H)*V_bad_b_p)
        V_hat_bad_itp = Akima(a_grid_pos, V_hat_bad)

        for a_i in 1:a_size_pos
            a = a_grid_pos[a_i]
            CoH = exp(e) + (1+r_d)*a

            # identify the optimal regions with discrete gridpoints
            V_bad_all = u_func.(CoH .- a_grid_pos, σ) .+ V_hat_bad
            V_bad_max = maximum(V_bad_all)
            V_bad_max_ind = findall(V_bad_all .== V_bad_max)[1]

            # solve it with interpolation method
            object_bad(a_p) = -(u_func(CoH - a_p[1], σ) + V_hat_bad_itp(a_p[1]))
            function gradient_bad!(G, a_p)
                G[1] = derivative(object_bad, a_p[1])
            end
            if a_grid_pos[V_bad_max_ind] >= CoH
                initial_bad = CoH - 10^(-5)
            elseif a_grid_pos[V_bad_max_ind] <= 0.0
                initial_bad = 0.0 + 10^(-5)
            else
                initial_bad = a_grid_pos[V_bad_max_ind]
            end
            res_bad = optimize(object_bad, gradient_bad!, [0.0], [CoH], [initial_bad], Fminbox(GradientDescent()))

            # record results
            variables.V_bad[a_i,e_i,ν_i] = -Optim.minimum(res_bad)
            variables.policy_a_bad[a_i,e_i,ν_i] = Optim.minimizer(res_bad)[]
        end

        # good credit history and default
        variables.V_good_default[:,e_i,ν_i] .= u_func((1-ξ)*exp(e), σ) + V_hat_bad[1]

        # good credit history and repay
        V_good_g_p = (ν_p*V_good_p[:,:,1] + (1-ν_p)*V_good_p[:,:,2])*e_Γ[e_i,:]
        V_hat_good = (ν*β_H)*V_good_g_p
        V_hat_good_itp = Akima(a_grid, V_hat_good)


        q = q_p[:,e_i,ν_i]
        qa = q .* a_grid
        qa_itp = Akima(a_grid, qa)

        for a_i in 1:a_size
            a = a_grid[a_i]
            CoH = exp(e) + (1+r_d*(a>0))*a

            # identify the optimal regions with discrete gridpoints
            V_good_all = u_func.(CoH .- qa, σ) .+ V_hat_good
            V_good_max = maximum(V_good_all)
            V_good_max_ind = findall(V_good_all .== V_good_max)[1]

            # solve it with interpolation method
            object_good(a_p) = -(u_func(CoH - qa_itp(a_p[1]), σ) + V_hat_good_itp(a_p[1]))
            function gradient_good!(G, a_p)
                G[1] = derivative(object_good, a_p[1])
            end

            if a_grid[V_good_max_ind] >= CoH
                initial_good = CoH - 10^(-5)
            elseif a_grid[V_good_max_ind] <= a_grid[1]
                initial_good = a_grid[1] + 10^(-5)
            else
                initial_good = a_grid[V_good_max_ind]
            end
            res_good = optimize(object_good, gradient_good!, [a_grid[1]], [CoH], [initial_good], Fminbox(GradientDescent()))

            # record results
            variables.V_good_repay[a_i,e_i,ν_i] = -Optim.minimum(res_good)
            if variables.V_good_default[a_i,e_i,ν_i] > variables.V_good_repay[a_i,e_i,ν_i]
                variables.V_good[a_i,e_i,ν_i] = variables.V_good_default[a_i,e_i,ν_i]
                variables.policy_a_good[a_i,e_i,ν_i] = 0.0
                variables.policy_d_good[a_i,e_i,ν_i] = 1.0
            else
                variables.V_good[a_i,e_i,ν_i] = variables.V_good_repay[a_i,e_i,ν_i]
                variables.policy_a_good[a_i,e_i,ν_i] = Optim.minimizer(res_good)[]
                variables.policy_d_good[a_i,e_i,ν_i] = 0.0
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

        V_diff_1 = variables.V_good_repay[a_p_i,:,1] .- variables.V_good_default[a_p_i,:,1]
        if all(V_diff_1 .> 0)
            e_p_thres_1 = -Inf
        elseif all(V_diff_1 .< 0)
            e_p_thres_1 = Inf
        else
            e_p_lower_1 = e_grid[maximum(findall(V_diff_1 .<= 0.0))]
            e_p_upper_1 = e_grid[minimum(findall(V_diff_1 .>= 0.0))]
            V_diff_1_itp = Akima(e_grid, V_diff_1)
            object_V_diff_1(e_p) = V_diff_1_itp(e_p)
            e_p_thres_1 = find_zero(object_V_diff_1, (e_p_lower_1, e_p_upper_1), Bisection())
        end

        V_diff_2 = variables.V_good_repay[a_p_i,:,2] .- variables.V_good_default[a_p_i,:,2]
        if all(V_diff_2 .> 0)
            e_p_thres_2 = -Inf
        elseif all(V_diff_2 .< 0)
            e_p_thres_2 = Inf
        else
            e_p_lower_2 = e_grid[maximum(findall(V_diff_2 .<= 0.0))]
            e_p_upper_2 = e_grid[minimum(findall(V_diff_2 .>= 0.0))]
            V_diff_2_itp = Akima(e_grid, V_diff_2)
            object_V_diff_2(e_p) = V_diff_2_itp(e_p)
            e_p_thres_2 = find_zero(object_V_diff_2, (e_p_lower_2, e_p_upper_2), Bisection())
        end

        for x_i in 1:x_size
            e, ν = x_grid[x_i,:]
            e_i, ν_i = x_ind[x_i,:]
            dist = Normal(0.0,1.0)

            default_prob_1 = cdf(dist, (e_p_thres_1-e_ρ*e)/e_σ)
            default_prob_2 = cdf(dist, (e_p_thres_2-e_ρ*e)/e_σ)
            repay_prob = ν_p*(1.0-default_prob_1) + (1.0-ν_p)*(1.0-default_prob_2)

            garnishment_1 = cdf(dist, (e_p_thres_1-(e_ρ*e+e_σ^2))/e_σ)
            garnishment_2 = cdf(dist, (e_p_thres_2-(e_ρ*e+e_σ^2))/e_σ)
            garnishment_rate = (ξ/-a_grid[a_p_i])*exp(e_ρ*e + (e_σ^2/2.0))*(ν_p*garnishment_1 + (1.0-ν_p)*garnishment_2)

            # q_update[a_p_i,e_i,ν_i] = repay_prob / (1.0+r_d+r_ld)
            variables.prob_default[a_p_i,e_i,ν_i] = 1.0 - repay_prob
            q_update[a_p_i,e_i,ν_i] = (repay_prob+garnishment_rate) / (1.0+r_d+r_ld)
        end

    end
    variables.q[1:a_size_neg_nozero,:,:] = Δ*q_update + (1-Δ)*q_p[1:a_size_neg_nozero,:,:]
end

function household_func!(
    variables::mut_var,
    parameters::NamedTuple;
    tol = tol_h,
    iter_max = iter_max,
    oneloop_algo::Integer = 1
    )

    if oneloop_algo == 1

        # initialize the iteration number and criterion
        iter = 0
        crit = Inf
        prog = ProgressThresh(tol, "Solving household's maximization (one loop): ")

        # initialize the next-period value functions
        V_good_p = similar(variables.V_good)
        V_bad_p = similar(variables.V_bad)
        q_p = similar(variables.q)

        while crit > tol && iter < iter_max

            # println("iter = $iter")
            # copy the current value functions to the pre-specified containers
            copyto!(V_good_p, variables.V_good)
            copyto!(V_bad_p, variables.V_bad)
            copyto!(q_p, variables.q)

            # update value function
            value_func!(V_good_p, V_bad_p, q_p, variables, parameters)

            # update price, its derivative, and size of bond
            price_func!(q_p, variables, parameters)

            # check convergence
            crit = max(norm(variables.V_good .- V_good_p, Inf), norm(variables.V_bad .- V_bad_p, Inf), norm(variables.q .- q_p, Inf))

            # report preogress
            ProgressMeter.update!(prog, crit)

            # update the iteration number
            iter += 1
        end

    else

        # initialize the iteration number and criterion
        iter_p = 0
        crit_p = Inf
        prog_p = ProgressThresh(tol, "Update pricing function: ")

        # initialize the next-period pricing function
        q_p = similar(variables.q)

        while crit_p > tol && iter_p < iter_max

            # copy the current pricing functions to the pre-specified container
            copyto!(q_p, variables.q)

            # initialize the iteration number and criterion
            iter_v = 0
            crit_v = Inf
            prog_v = ProgressThresh(tol, "Solving value functions: ")

            # initialize the next-period value functions
            V_good_p = similar(variables.V_good)
            V_bad_p = similar(variables.V_bad)

            while crit_v > tol && iter_v < iter_max

                # copy the current value functions to the pre-specified containers
                copyto!(V_good_p, variables.V_good)
                copyto!(V_bad_p, variables.V_bad)

                # update value function
                value_func!(V_good_p, V_bad_p, q_p, variables, parameters)

                # check convergence
                crit_v = max(norm(variables.V_good .- V_good_p, Inf), norm(variables.V_bad .- V_bad_p, Inf))

                # report preogress
                ProgressMeter.update!(prog_v, crit_v)

                # update the iteration number
                iter_v += 1
            end

            # update price, its derivative, and size of bond
            price_func!(q_p, variables, parameters)

            # check convergence
            crit_p = norm(variables.q .- q_p, Inf)

            # report preogress
            ProgressMeter.update!(prog_p, crit_p)

            # update the iteration number
            iter_p += 1
        end
    end
end

function density_func!(
    variables::mut_var,
    parameters::NamedTuple;
    tol = tol_μ,
    iter_max = iter_max
    )

    @unpack ψ_H = parameters
    @unpack x_size, x_ind, e_Γ, ν_Γ, e_size, ν_size = parameters
    @unpack a_size, a_grid, a_size_pos, a_grid_pos = parameters
    @unpack a_size_μ, a_grid_μ, a_size_pos_μ, a_grid_pos_μ, a_ind_zero_μ = parameters

    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving invariant density: ")

    μ_good_p = similar(variables.μ_good)
    μ_bad_p = similar(variables.μ_bad)

    while crit > tol && iter < iter_max

        copyto!(μ_good_p, variables.μ_good)
        copyto!(μ_bad_p, variables.μ_bad)

        variables.μ_good .= 0.0
        variables.μ_bad .= 0.0

        μ_good_temp = zeros(a_size_μ, e_size, ν_size, e_size*ν_size)
        μ_bad_temp = zeros(a_size_pos_μ, e_size, ν_size,  e_size*ν_size)

        # @showprogress 1 "Iterating good credit history: "
        Threads.@threads for x_i in 1:x_size

            e_i, ν_i = x_ind[x_i,:]

            # good credit history
            policy_a_good_itp = Akima(a_grid, variables.policy_a_good[:,e_i,ν_i])

            if any(variables.policy_d_good[:,e_i,ν_i] .== 1.0)
                a_ind_default = findall(variables.policy_d_good[:,e_i,ν_i] .== 1.0)[end]
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

                    a_p = clamp(policy_a_good_itp(a_grid_μ[a_i]), a_grid[1], a_grid[end])
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
                        μ_good_temp[ind_lower_a_p,e_p_i,ν_p_i,x_i] += e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_lower * μ_good_p[a_i,e_i,ν_i]
                        μ_good_temp[ind_upper_a_p,e_p_i,ν_p_i,x_i] += e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_upper * μ_good_p[a_i,e_i,ν_i]
                        # variables.μ_good[ind_lower_a_p,e_p_i,ν_p_i] += e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_lower * μ_good_p[a_i,e_i,ν_i]
                        # variables.μ_good[ind_upper_a_p,e_p_i,ν_p_i] += e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_upper * μ_good_p[a_i,e_i,ν_i]
                    end

                # good -> bad
                else
                    for x_p_i in 1:x_size
                        e_p_i, ν_p_i = x_ind[x_p_i,:]
                        μ_bad_temp[1,e_p_i,ν_p_i,x_i] += e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * μ_good_p[a_i,e_i,ν_i]
                        # variables.μ_bad[1,e_p_i,ν_p_i] += e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * μ_good_p[a_i,e_i,ν_i]
                    end
                end
            end

            # bad credit history
            policy_a_bad_itp = Akima(a_grid_pos, variables.policy_a_bad[:,e_i,ν_i])

            for a_i in 1:a_size_pos_μ

                a_p = clamp(policy_a_bad_itp(a_grid_pos_μ[a_i]), a_grid_pos[1], a_grid_pos[end])
                ind_lower_a_p = findall(a_grid_pos_μ .<= a_p)[end]
                ind_upper_a_p = findall(a_p .<= a_grid_pos_μ)[1]

                if ind_lower_a_p != ind_upper_a_p
                    a_lower_a_p = a_grid_pos_μ[ind_lower_a_p]
                    a_upper_a_p = a_grid_pos_μ[ind_upper_a_p]
                    weight_lower = (a_upper_a_p - a_p) / (a_upper_a_p - a_lower_a_p)
                    weight_upper = (a_p - a_lower_a_p) / (a_upper_a_p - a_lower_a_p)
                else
                    weight_lower = 0.5
                    weight_upper = 0.5
                end

                for x_p_i in 1:x_size

                    e_p_i, ν_p_i = x_ind[x_p_i,:]

                    # bad -> bad
                    μ_bad_temp[ind_lower_a_p,e_p_i,ν_p_i,x_i] += (1.0-ψ_H) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_lower * μ_bad_p[a_i,e_i,ν_i]
                    μ_bad_temp[ind_upper_a_p,e_p_i,ν_p_i,x_i] += (1.0-ψ_H) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_upper * μ_bad_p[a_i,e_i,ν_i]
                    # variables.μ_bad[ind_lower_a_p,e_p_i,ν_p_i] += (1.0-ψ_H) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_lower * μ_bad_p[a_i,e_i,ν_i]
                    # variables.μ_bad[ind_upper_a_p,e_p_i,ν_p_i] += (1.0-ψ_H) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_upper * μ_bad_p[a_i,e_i,ν_i]

                    # bad -> good
                    μ_good_temp[ind_lower_a_p+a_ind_zero_μ-1,e_p_i,ν_p_i,x_i] += ψ_H * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_lower * μ_bad_p[a_i,e_i,ν_i]
                    μ_good_temp[ind_upper_a_p+a_ind_zero_μ-1,e_p_i,ν_p_i,x_i] += ψ_H * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_upper * μ_bad_p[a_i,e_i,ν_i]
                    # variables.μ_good[ind_lower_a_p+a_ind_zero_μ-1,e_p_i,ν_p_i] += ψ_H * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_lower * μ_bad_p[a_i,e_i,ν_i]
                    # variables.μ_good[ind_upper_a_p+a_ind_zero_μ-1,e_p_i,ν_p_i] += ψ_H * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_upper * μ_bad_p[a_i,e_i,ν_i]
                end
            end
        end

        variables.μ_good .= dropdims(sum(μ_good_temp, dims=4), dims=4)
        variables.μ_bad.= dropdims(sum(μ_bad_temp, dims=4), dims=4)
        μ_sum = sum(variables.μ_good) + sum(variables.μ_bad)
        variables.μ_good .= variables.μ_good ./ μ_sum
        variables.μ_bad .= variables.μ_bad ./ μ_sum

        # check convergence
        crit = max(norm(variables.μ_good .- μ_good_p, Inf), norm(variables.μ_bad .- μ_bad_p, Inf))

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

    variables.aggregate_var .= 0.0

    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        qa_itp = Akima(a_grid, a_grid.*variables.q[:,e_i,ν_i])
        for a_μ_i in 1:(a_size_neg_μ-1)
            a_μ = a_grid_neg_μ[a_μ_i]
            variables.aggregate_var[1] += -(variables.μ_good[a_μ_i,e_i,ν_i] * qa_itp(a_μ))
        end
    end
    variables.aggregate_var[2] = sum(variables.μ_good[(a_ind_zero_μ+1):end,:,:].*repeat(a_grid_pos_μ[2:end],1,e_size,ν_size)) + sum(variables.μ_bad[2:end,:,:].*repeat(a_grid_pos_μ[2:end],1,e_size,ν_size))
    variables.aggregate_var[3] = variables.aggregate_var[1] - variables.aggregate_var[2]
    variables.aggregate_var[4] = variables.aggregate_var[1] / variables.aggregate_var[3]
end

function solve_func!(
    variables::mut_var,
    parameters::NamedTuple;
    tol_h = 1E-8,
    tol_μ = 1E-8,
    iter_max = 1E+5
    )

    # solve the household's problem (including price schemes)
    household_func!(variables, parameters; tol = tol_h, iter_max = iter_max)

    # update the cross-sectional distribution
    density_func!(variables, parameters; tol = tol_μ, iter_max = iter_max)

    # compute aggregate variables
    aggregate_func!(variables, parameters)

    ED = variables.aggregate_var[4] - parameters.LR

    data_spec = Any[#=1=# "Multiplier"              parameters.λ;
                    #=2=# "Leverage Ratio (Demand)" variables.aggregate_var[4];
                    #=3=# "Leverage Ratio (Supply)" parameters.LR;
                    #=4=# "Difference"              ED]

    pretty_table(data_spec, ["Name", "Value"];
                 alignment=[:l,:r],
                 formatters = ft_round(12),
                 body_hlines = [1,3])

    # save results
    V_good = variables.V_good
    V_bad = variables.V_bad
    q = variables.q
    μ_good = variables.μ_good
    μ_bad = variables.μ_bad
    @save "24092020_initial_values.bson" V_good V_bad q μ_good μ_bad

    return ED
end

parameters = para_func(;λ = 0.0)
variables = var_func(parameters)

println("Using Julia with $(Threads.nthreads()) threads....")
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
                #=14=# "Leverage Ratio (Supply)"            parameters.LR;
                #=15=# "Additional Opportunity Cost"        parameters.r_ld;
                #=16=# "Total Loans"                        variables.aggregate_var[1];
                #=17=# "Total Deposits"                     variables.aggregate_var[2];
                #=18=# "Net Worth"                          variables.aggregate_var[3];
                #=19=# "Leverage Ratio (Demand)"            variables.aggregate_var[4]]

hl_LR = Highlighter(f      = (data,i,j) -> i == 14 || i == 19,
                    crayon = Crayon(background = :light_blue))

pretty_table(data_spec, ["Name", "Value"];
             alignment=[:l,:r],
             formatters = ft_round(4),
             body_hlines = [7,9,15],
             highlighters = hl_LR)

#=
parameters = para_func()
para_targeted(x) = para_func(; λ = x)
solve_targeted(x) = solve_func!(var_func(para_targeted(x)), para_targeted(x))
# λ_lower, λ_upper = 0, 1-(parameters.β_B*parameters.ψ_B*(1+parameters.r_d))^(1/2)
# λ_optimal = find_zero(solve_targeted, (λ_lower, λ_upper), Bisection())
λ_optimal = find_zero(solve_targeted, (0.0431496070, 0.0431496073), Bisection())
λ_optimal = 0.04314960716505481

parameters = para_func(; λ = λ_optimal)
variables = var_func(parameters)
println("Using Julia with $(Threads.nthreads()) threads....")
solve_func!(variables, parameters)
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
