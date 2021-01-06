include("FLOWMath.jl")
using Main.FLOWMath: Akima, akima, interp2d
using LinearAlgebra
using ProgressMeter
using Parameters
using QuantEcon: rouwenhorst, tauchen, gridmake, MarkovChain, stationary_distributions
using Plots
# using LaTeXStrings
# using PlotThemes
using PrettyTables
using Roots
using Optim
using Calculus: derivative
using Distributions
# using SparseArrays
using BSON: @save, @load
# using UnicodePlots: spy
# using Expectations
using QuadGK: quadgk

function para_func(;
    β_H::Real = 0.96,           # discount factor (households)
    β_B::Real = 0.96,           # discount factor (banks)
    σ::Real = 2.00,             # CRRA coefficient
    η::Real = 0.40,             # garnishment rate
    ξ_bar::Real = 0.1,          # upper bound of random utility cost
    δ::Real = 0.08,             # depreciation rate
    α::Real = 0.33,             # capital share
    K2Y::Real = 3.0,            # capital to output ratio
    ψ::Real = 0.95,             # bank's survival rate
    λ::Real = 0.00,             # multiplier of incentive constraint
    θ::Real = 0.40,             # diverting fraction
    e_ρ::Real = 0.95,           # AR(1) of endowment shock
    e_σ::Real = 0.10,           # s.d. of endowment shock
    e_size::Integer = 9,        # number of endowment shock
    ν_s::Real = 0.95,           # scale of patience
    ν_p::Real = 0.10,           # probability of patience
    ν_size::Integer = 2,        # number of preference shock
    a_min::Real = -1.5,         # min of asset holding
    a_max::Real = 350.0,        # max of asset holding
    a_size_neg::Integer = 151,  # number of grid of negative asset holding for VFI
    a_size_pos::Integer = 51,   # number of grid of positive asset holding for VFI
    a_degree::Integer = 3,      # curvature of the positive asset gridpoints
    μ_scale::Integer = 7        # scale governing the number of grids in computing density
    )
    """
    contruct an immutable object containg all paramters
    """

    # endowment shock
    e_MC = tauchen(e_size, e_ρ, e_σ, 0.0, 3)
    e_Γ = e_MC.p
    e_grid = collect(e_MC.state_values)
    e_SD = stationary_distributions(e_MC)[]
    e_SS = sum(e_SD .* e_grid)

    # preference schock
    ν_grid = [ν_s, 1.0]
    ν_Γ = repeat([ν_p 1.0-ν_p], ν_size, 1)

    # idiosyncratic transition matrix
    x_Γ = kron(ν_Γ, e_Γ)
    x_grid = gridmake(e_grid, ν_grid)
    x_ind = gridmake(1:e_size, 1:ν_size)
    x_size = e_size*ν_size

    # asset holding grid for VFI
    # a_min = -floor(exp(e_grid[end])/2-e_σ, digits=2)
    # a_size_neg = convert(Int, -a_min*100+1)
    a_grid_neg = collect(range(a_min, 0.0, length = a_size_neg))
    a_grid_pos = ((range(0.0, stop = a_size_pos-1, length = a_size_pos)/(a_size_pos-1)).^a_degree)*a_max
    a_grid = cat(a_grid_neg, a_grid_pos[2:end], dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero, a_grid)[]

    # asset holding grid for μ
    # a_size_neg_μ = convert(Int, (a_size_neg-1)*μ_scale+1)
    a_size_neg_μ = convert(Int, a_size_neg)
    a_grid_neg_μ = collect(range(a_min, 0.0, length = a_size_neg_μ))
    a_size_pos_μ = convert(Int, (a_size_pos-1)*μ_scale+1)
    a_grid_pos_μ = collect(range(0.0, a_max, length = a_size_pos_μ))
    a_grid_μ = cat(a_grid_neg_μ, a_grid_pos_μ[2:end], dims = 1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(iszero, a_grid_μ)[]

    # compute equilibrium prices and quantities
    i = (α/K2Y) - δ
    γ = (β_B*(1.0-ψ)*(1.0+i)) / ((1.0-λ)-β_B*ψ*(1.0+i))
    Λ = β_B*(1.0-ψ+ψ*γ)
    LR = γ/θ
    AD = LR/(LR-1.0)
    r_lp = λ*θ/Λ
    r_k = i + r_lp
    K = exp(e_SS)*(α/(r_k+δ))^(1.0/(1.0-α))
    w = (1.0-α)*(K^α)*(exp(e_SS)^(-α))

    # return values
    return (β_H = β_H, β_B = β_B, σ = σ, η = η, ξ_bar = ξ_bar, δ = δ, α = α,
            K2Y = K2Y, ψ = ψ, λ = λ, θ = θ, a_degree = a_degree, μ_scale = μ_scale,
            e_ρ = e_ρ, e_σ = e_σ, e_size = e_size, e_Γ = e_Γ, e_grid = e_grid, e_SS = e_SS,
            ν_s = ν_s, ν_p = ν_p, ν_size = ν_size, ν_Γ = ν_Γ, ν_grid = ν_grid,
            x_Γ = x_Γ, x_grid = x_grid, x_ind = x_ind, x_size = x_size,
            a_grid = a_grid, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos,
            a_size = a_size, a_size_neg = a_size_neg, a_size_pos = a_size_pos,
            a_ind_zero = a_ind_zero,
            a_grid_μ = a_grid_μ, a_grid_neg_μ = a_grid_neg_μ, a_grid_pos_μ = a_grid_pos_μ,
            a_size_μ = a_size_μ, a_size_neg_μ = a_size_neg_μ, a_size_pos_μ = a_size_pos_μ,
            a_ind_zero_μ = a_ind_zero_μ,
            i = i, γ = γ, Λ = Λ, LR = LR, AD = AD, r_lp = r_lp, r_k = r_k, K = K, w = w)
end

mutable struct MutableVariables
    """
    construct a type for mutable variables
    """
    V::Array{Float64,3}
    V_nd::Array{Float64,3}
    V_d::Array{Float64,2}
    policy_a::Array{Float64,3}
    policy_d::Array{Float64,3}
    q::Array{Float64,2}
    μ::Array{Float64,3}
    aggregate_var::Array{Float64,1}
end

function var_func(
    parameters::NamedTuple;
    load_initial_values::Integer = 0
    )
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack a_size, a_size_neg, a_size_μ, e_size, ν_size, i, r_lp = parameters

    if load_initial_values == 1
        @load "18122020_initial_values.bson" V q μ

        # define value functions
        V_nd = zeros(a_size, e_size, ν_size)
        V_d = zeros(e_size, ν_size)

        # define policy functions
        policy_a = zeros(a_size, e_size, ν_size)
        policy_d = zeros(a_size, e_size, ν_size)
    else
        # define value functions
        V = zeros(a_size, e_size, ν_size)
        V_nd = zeros(a_size, e_size, ν_size)
        V_d = zeros(e_size, ν_size)

        # define policy functions
        policy_a = zeros(a_size, e_size, ν_size)
        policy_d = zeros(a_size, e_size, ν_size)

        # define pricing function
        q = ones(a_size_neg, e_size) ./ (1.0 + i + r_lp)

        # define the type distribution and its transition matrix
        μ_size = a_size_μ*e_size*ν_size
        μ = ones(a_size_μ, e_size, ν_size) ./ μ_size
    end

    # define aggregate variables
    aggregate_var = zeros(8)

    # return outputs
    variables = MutableVariables(V, V_nd, V_d,
                                 policy_a, policy_d,
                                 q, μ, aggregate_var)
    return variables
end

function u_func(
    c::Real,
    σ::Real)
    """
    compute utility of CRRA utility function with coefficient σ
    """
    if c > 0
        return σ == 1 ? log(c) : 1 / ((1-σ)*c^(σ-1))
    else
        return -10^8
        # return typemin(eltype(c))
    end
end

function value_func!(
    V_p::Array{Float64,3},
    q_p::Array{Float64,2},
    variables::MutableVariables,
    parameters::NamedTuple
    )
    """
    update value functions
    """

    @unpack β_H, σ, η, ξ_bar, i, w = parameters
    @unpack a_size, a_grid, a_grid_neg, a_grid_pos, a_ind_zero = parameters
    @unpack x_size, x_grid, x_ind, ν_Γ, e_Γ = parameters

    Threads.@threads for x_i in 1:x_size

        e, ν = x_grid[x_i,:]
        e_i, ν_i = x_ind[x_i,:]

        # compute the next-period expected value funtion
        V_expt_p = (ν_Γ[ν_i,1]*V_p[:,:,1] + ν_Γ[ν_i,2]*V_p[:,:,2])*e_Γ[e_i,:]
        V_hat = (ν*β_H)*V_expt_p
        V_hat_itp = Akima(a_grid, V_hat)

        # compute defaulting value
        variables.V_d[e_i,ν_i] = u_func((1-η)*w*exp(e), σ) + V_hat[a_ind_zero]

        # compute non-defaulting value
        q = q_p[:,e_i]
        qa = [q.*a_grid_neg; a_grid_pos[2:end]]
        qa_itp = Akima(a_grid, qa)

        # Search initial value with discrete gridpoints
        qa_ind = argmin(qa)

        # set up objective function and its gradient
        object_rbl(a_p) = qa_itp(a_p[1])
        function gradient_rbl!(G, a_p)
            G[1] = derivative(object_rbl, a_p[1])
        end

        # make sure the initial value is not on the boundaries
        if a_grid[qa_ind] <= a_grid[1]
            initial = a_grid[1] + 10^(-6)
        else
            initial = a_grid[qa_ind]
        end

        # find the risky borrowing limit
        res_rbl = optimize(object_rbl, gradient_rbl!,
                           [a_grid[1]], [a_grid[a_ind_zero]],
                           [initial],
                           Fminbox(GradientDescent()))
        rbl = Optim.minimizer(res_rbl)[]

        # compute non-defaulting value
        Threads.@threads for a_i in 1:a_size

            a = a_grid[a_i]
            CoH = w*exp(e) + (1+i*(a>0))*a

            if CoH - qa_itp(rbl) >= 0.0

                # Search initial value with discrete gridpoints
                V_all = u_func.(CoH .- qa, σ) .+ V_hat
                V_max_ind = argmax(V_all)

                # set up objective function and its gradient
                object_nd(a_p) = -(u_func(CoH - qa_itp(a_p[1]), σ) + V_hat_itp(a_p[1]))
                function gradient_nd!(G, a_p)
                    G[1] = derivative(object_nd, a_p[1])
                end

                # make sure the initial value is not on the boundaries
                if a_grid[V_max_ind] >= CoH
                    initial = CoH - 10^(-6)
                elseif a_grid[V_max_ind] <= a_grid[1]
                    initial = a_grid[1] + 10^(-6)
                else
                    initial = a_grid[V_max_ind]
                end

                # find the optimal asset holding
                res_nd = optimize(object_nd, gradient_nd!,
                                  [a_grid[1]], [CoH],
                                  [initial],
                                  Fminbox(GradientDescent()))

                # record results
                variables.V_nd[a_i,e_i,ν_i] = -Optim.minimum(res_nd)
                variables.policy_a[a_i,e_i,ν_i] = Optim.minimizer(res_nd)[]

            else

                # record results
                variables.V_nd[a_i,e_i,ν_i] = u_func(CoH - qa_itp(rbl), σ) + V_hat_itp(rbl)
                variables.policy_a[a_i,e_i,ν_i] = rbl
            end
        end

        # compute cutoff
        ξ_star = variables.V_d[e_i,ν_i] .- variables.V_nd[:,e_i,ν_i]
        clamp!(ξ_star, 0.0, ξ_bar)
        F_ξ_star = ξ_star ./ ξ_bar

        # determine value function
        variables.V[:,e_i,ν_i] .= -(ξ_star.^2)./(2.0*ξ_bar) .+ F_ξ_star.*variables.V_d[e_i,ν_i] .+ (1.0 .- F_ξ_star).*variables.V_nd[:,e_i,ν_i]
        variables.policy_d[:,e_i,ν_i] .= F_ξ_star
    end
end

function find_threshold_func(
    V_nd::Array{Float64,1},
    V_d::Array{Float64,1},
    e_grid::Array{Float64,1},
    cutoff_value::Real;
    range::Real
    )
    """
    compute the threshold below which households file for bankruptcy
    """

    V_diff = V_nd .- V_d .+ cutoff_value
    V_nd_itp = Akima(e_grid, V_nd)
    V_d_itp = Akima(e_grid, V_d)
    V_diff_itp(x) = V_nd_itp(x) - V_d_itp(x) + cutoff_value

    if all(V_diff .> 0.0)
        e_p_lower = e_grid[1] - range
        if V_diff_itp(e_p_lower) > 0.0
            e_p_thres = e_p_lower
        else
            e_p_thres = find_zero(e_p->V_diff_itp(e_p), (e_p_lower, e_grid[1]), Bisection())
        end
    elseif all(V_diff .< 0.0)
        e_p_upper = e_grid[end] + range
        if V_diff_itp(e_p_upper) < 0.0
            e_p_thres = e_p_upper
        else
            e_p_thres = find_zero(e_p->V_diff_itp(e_p), (e_grid[end], e_p_upper), Bisection())
        end
    else
        e_p_lower = e_grid[findall(V_diff .<= 0.0)[end]]
        e_p_upper = e_grid[findall(V_diff .>= 0.0)[1]]
        e_p_thres = find_zero(e_p->V_diff_itp(e_p), (e_p_lower, e_p_upper), Bisection())
    end

    return e_p_thres
end

function price_func!(
    q_p::Array{Float64,2},
    variables::MutableVariables,
    parameters::NamedTuple;
    Δ::Real = 0.7
    )
    """
    update price function
    """

    @unpack ξ_bar, i, r_lp, a_size_neg, e_size, e_grid, e_ρ, e_σ, ν_p = parameters

    # create the container
    q_update = zeros(a_size_neg, e_size)

    Threads.@threads for a_p_i in 1:a_size_neg

        # compute defaulting threshold for (im)patient households
        e_p_thres_ξ_bar_1 = find_threshold_func(variables.V_nd[a_p_i,:,1], variables.V_d[:,1], e_grid, ξ_bar, range = 8*e_σ)
        e_p_thres_ξ_bar_2 = find_threshold_func(variables.V_nd[a_p_i,:,2], variables.V_d[:,2], e_grid, ξ_bar, range = 8*e_σ)

        # create default policy functions
        V_nd_1_itp = Akima(e_grid, variables.V_nd[a_p_i,:,1])
        V_d_1_itp = Akima(e_grid, variables.V_d[:,1])
        V_diff_1_itp(x) = V_d_1_itp(x) - V_nd_1_itp(x)
        default_policy_1(x) = clamp(V_diff_1_itp(x), 0.0, ξ_bar) / ξ_bar

        V_nd_2_itp = Akima(e_grid, variables.V_nd[a_p_i,:,2])
        V_d_2_itp = Akima(e_grid, variables.V_d[:,2])
        V_diff_2_itp(x) = V_d_2_itp(x) - V_nd_2_itp(x)
        default_policy_2(x) = clamp(V_diff_2_itp(x), 0.0, ξ_bar) / ξ_bar

        for e_i in 1:e_size
            e = e_grid[e_i]
            dist = Normal(e_ρ*e, e_σ)

            # compute default probability for (im)patient households
            kernel_1(x) = (1-default_policy_1(x))*pdf(dist, x)
            kernel_2(x) = (1-default_policy_2(x))*pdf(dist, x)

            if e_p_thres_ξ_bar_1 == Inf
                repay_prob_1 = 0.0
            elseif e_p_thres_ξ_bar_1 == -Inf
                repay_prob_1 = 1.0
            else
                repay_prob_1, err1 = quadgk(x -> kernel_1(x), e_p_thres_ξ_bar_1, Inf) #, order=100, rtol=1E-10
            end

            if e_p_thres_ξ_bar_2 == Inf
                repay_prob_2 = 0.0
            elseif e_p_thres_ξ_bar_2 == -Inf
                repay_prob_2 = 1.0
            else
                repay_prob_2, err2 = quadgk(x -> kernel_1(x), e_p_thres_ξ_bar_2, Inf) #, order=100, rtol=1E-10
            end

            repay_prob = ν_p*repay_prob_1 + (1.0-ν_p)*repay_prob_2

            # update bond price
            q_update[a_p_i,e_i] = clamp(repay_prob, 0.0, 1.0) / (1.0 + i + r_lp)
        end
    end

    # clamp!(q_update, 0.0, 1.0/(1.0 + i + r_lp))
    variables.q = Δ*q_update + (1-Δ)*q_p
end

function household_func!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol::Real = tol_h,
    iter_max::Real = iter_max
    )
    """
    update value and price functions simultaneously
    """

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving household's maximization (one loop): ")

    # initialize the next-period value functions
    V_p = similar(variables.V)
    q_p = similar(variables.q)

    while crit > tol && iter < iter_max

        # copy the current value functions to the pre-specified containers
        copyto!(V_p, variables.V)
        copyto!(q_p, variables.q)

        # update value function
        value_func!(V_p, q_p, variables, parameters)

        # update price function
        price_func!(q_p, variables, parameters)

        # check convergence
        crit = max(norm(variables.V .- V_p, Inf), norm(variables.q .- q_p, Inf))
        # println("V_diff = $(norm(variables.V .- V_p, Inf)) and q_diff = $(norm(variables.q .- q_p, Inf))")
        # println("$(argmax(variables.V .- V_p))")
        # println("$(argmax(variables.q .- q_p))")

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end

function density_func!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol = tol_μ,
    iter_max = iter_max
    )
    """
    update the cross-sectional distribution
    """

    @unpack ξ_bar, x_size, x_ind, e_Γ, ν_Γ, a_grid, a_size_μ, a_grid_μ, a_ind_zero_μ = parameters

    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving invariant density: ")

    # copy the previous values
    μ_p = similar(variables.μ)

    while crit > tol && iter < iter_max

        copyto!(μ_p, variables.μ)
        variables.μ .= 0.0

        for x_i in 1:x_size
            e_i, ν_i = x_ind[x_i,:]

            # interpolate decision rules
            policy_a_itp = Akima(a_grid, variables.policy_a[:,e_i,ν_i])
            V_nd_itp = Akima(a_grid, variables.V_nd[:,e_i,ν_i])
            V_diff_itp(x) = variables.V_d[e_i,ν_i] - V_nd_itp(x)
            policy_d_itp(x) = clamp(V_diff_itp(x), 0.0, ξ_bar) / ξ_bar

            # loop over the dimension of asset holding
            for a_i in 1:a_size_μ

                # locate it in the original grid
                a_μ = a_grid_μ[a_i]
                a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])
                ind_lower_a_p = findall(a_grid_μ .<= a_p)[end]
                ind_upper_a_p = findall(a_p .<= a_grid_μ)[1]

                # compute weights
                if ind_lower_a_p != ind_upper_a_p
                    a_lower_a_p = a_grid_μ[ind_lower_a_p]
                    a_upper_a_p = a_grid_μ[ind_upper_a_p]
                    weight_lower = (a_upper_a_p - a_p) / (a_upper_a_p - a_lower_a_p)
                    weight_upper = (a_p - a_lower_a_p) / (a_upper_a_p - a_lower_a_p)
                else
                    weight_lower = 0.5
                    weight_upper = 0.5
                end

                # loop over the dimension of exogenous individual states
                for x_p_i in 1:x_size
                    e_p_i, ν_p_i = x_ind[x_p_i,:]

                    # update the values
                    variables.μ[ind_lower_a_p,e_p_i,ν_p_i] += (1.0-policy_d_itp(a_μ)) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_lower * μ_p[a_i,e_i,ν_i]
                    variables.μ[ind_upper_a_p,e_p_i,ν_p_i] += (1.0-policy_d_itp(a_μ)) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_upper * μ_p[a_i,e_i,ν_i]
                    variables.μ[a_ind_zero_μ,e_p_i,ν_p_i] += policy_d_itp(a_μ) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * μ_p[a_i,e_i,ν_i]
                end
            end
        end
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
    variables::MutableVariables,
    parameters::NamedTuple
    )
    """
    compute aggregate variables
    """

    @unpack x_size, x_ind, x_grid, e_size, ν_size, a_grid, a_grid_neg, a_grid_pos = parameters
    @unpack a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ, a_grid_μ, a_size_μ = parameters
    @unpack K, ξ_bar = parameters

    # total loans and deposits
    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]

        # asset holding policy function
        policy_a_itp = Akima(a_grid, variables.policy_a[:,e_i,ν_i])

        # loan size function
        q = variables.q[:,e_i]
        qa = [q.*a_grid_neg; a_grid_pos[2:end]]
        qa_itp = Akima(a_grid, qa)

        # default policy function
        V_nd_itp = Akima(a_grid, variables.V_nd[:,e_i,ν_i])
        V_diff_itp(x) = variables.V_d[e_i,ν_i] - V_nd_itp(x)
        policy_d_itp(x) = clamp(V_diff_itp(x), 0.0, ξ_bar) / ξ_bar

        for a_μ_i in 1:a_size_μ
            a_μ = a_grid_μ[a_μ_i]
            a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])
            if a_p < 0.0
                # total loans
                variables.aggregate_var[1] += -(variables.μ[a_μ_i,e_i,ν_i] * (1.0-policy_d_itp(a_μ)) * qa_itp(a_p))
            else
                # total deposits
                variables.aggregate_var[2] += (variables.μ[a_μ_i,e_i,ν_i] * (1.0-policy_d_itp(a_μ)) * a_p)
            end
        end
    end

    # total deposits
    # variables.aggregate_var[2] = sum(variables.μ[(a_ind_zero_μ+1):end,:,:].*repeat(a_grid_pos_μ[2:end],1,e_size,ν_size))

    # net worth
    variables.aggregate_var[3] = (K + variables.aggregate_var[1]) - variables.aggregate_var[2]

    # asset-to-debt ratio (or loan-to-deposit ratio)
    variables.aggregate_var[4] = (K + variables.aggregate_var[1]) / variables.aggregate_var[2]

    # share of defaulters
    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        # policy_d_itp = Akima(a_grid, variables.policy_d[:,e_i,ν_i])
        V_nd_itp = Akima(a_grid, variables.V_nd[:,e_i,ν_i])
        V_diff_itp(x) = variables.V_d[e_i,ν_i] - V_nd_itp(x)
        policy_d_itp(x) = clamp(V_diff_itp(x), 0.0, ξ_bar) / ξ_bar
        for a_μ_i in 1:a_size_neg_μ
            a_μ = a_grid_neg_μ[a_μ_i]
            variables.aggregate_var[5] += (variables.μ[a_μ_i,e_i,ν_i] * policy_d_itp(a_μ))
        end
    end

    # share in debt
    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        for a_μ_i in 1:(a_size_neg_μ-1)
            a_μ = a_grid_neg_μ[a_μ_i]
            variables.aggregate_var[6] += variables.μ[a_μ_i,e_i,ν_i]
        end
    end

    # debt-to-income ratio
    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        e, ν = x_grid[x_i,:]
        for a_μ_i in 1:(a_size_neg_μ-1)
            a_μ = a_grid_neg_μ[a_μ_i]
            variables.aggregate_var[7] += variables.μ[a_μ_i,e_i,ν_i] * (-a_μ/exp(e))
        end
    end

    # average loan rate
    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]

        # asset holding policy function
        policy_a_itp = Akima(a_grid, variables.policy_a[:,e_i,ν_i])

        # loan size function
        q = variables.q[:,e_i]
        qa = [q.*a_grid_neg; a_grid_pos[2:end]]
        qa_itp = Akima(a_grid, qa)

        # default policy function
        V_nd_itp = Akima(a_grid, variables.V_nd[:,e_i,ν_i])
        V_diff_itp(x) = variables.V_d[e_i,ν_i] - V_nd_itp(x)
        policy_d_itp(x) = clamp(V_diff_itp(x), 0.0, ξ_bar) / ξ_bar

        for a_μ_i in 1:a_size_μ
            a_μ = a_grid_μ[a_μ_i]
            a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])
            if a_p < 0.0
                variables.aggregate_var[8] += (variables.μ[a_μ_i,e_i,ν_i] * (1.0-policy_d_itp(a_μ)) * (a_p/qa_itp(a_p)))
            end
        end
    end
end

function solve_func!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol_h::Real = 1E-6,
    tol_μ::Real = 1E-8,
    iter_max::Real = 1E+3
    )

    # solve the household's problem (including price schemes)
    household_func!(variables, parameters; tol = tol_h, iter_max = iter_max)

    # update the cross-sectional distribution
    density_func!(variables, parameters; tol = tol_μ, iter_max = iter_max)

    # compute aggregate variables
    aggregate_func!(variables, parameters)

    ED = variables.aggregate_var[4] - parameters.AD

    data_spec = Any[#=1=# "Garnishment Rate"             parameters.η;
                    #=2=# "Multiplier"                   parameters.λ;
                    #=3=# "Asset-to-Debt Ratio (Demand)" variables.aggregate_var[4];
                    #=4=# "Asset-to-Debt Ratio (Supply)" parameters.AD;
                    #=5=# "Difference"                   ED]

    pretty_table(data_spec, ["Name", "Value"];
                 alignment=[:l,:r],
                 formatters = ft_round(8),
                 body_hlines = [2,4])

    # save results
    V = variables.V
    q = variables.q
    μ = variables.μ
    @save "18122020_initial_values.bson" V q μ

    return ED
end

function λ_optimal_func(
    η::Real,
    a_min::Real;
    λ_min_adhoc::Real = -Inf,
    λ_max_adhoc::Real = Inf,
    tol::Real = 1E-6,
    iter_max::Real = 1E+3
    )
    """
    solve for the optimal multiplier
    """

    # compute the associated number of gridpoints fro negative asset
    a_size_neg = convert(Int, 1-a_min*200)

    # check the case of λ_min = 0.0
    λ_min = 0.0
    parameters_λ_min = para_func(η = η, a_min = a_min, a_size_neg = a_size_neg, λ = λ_min)
    variables_λ_min = var_func(parameters_λ_min, load_initial_values = 0)
    ED_λ_min = solve_func!(variables_λ_min, parameters_λ_min)
    if ED_λ_min > 0.0
        return parameters_λ_min, variables_λ_min, parameters_λ_min, variables_λ_min
    end

    # check the case of λ_max = 1-(β*ψ*(1+i))^(1/2)
    λ_max = 1.0 - (parameters_λ_min.β_B*parameters_λ_min.ψ*(1+parameters_λ_min.i))^(1/2)
    parameters_λ_max = para_func(η = η, a_min = a_min, a_size_neg = a_size_neg, λ = λ_max)
    variables_λ_max = var_func(parameters_λ_max, load_initial_values = 0)
    ED_λ_max = solve_func!(variables_λ_max, parameters_λ_max)
    if ED_λ_max < 0.0
        return parameters_λ_max, variables_λ_max, parameters_λ_max, variables_λ_max # solution doesn't exist!!!
    end

    # fing the optimal λ using bisection
    λ_lower = max(λ_min_adhoc, λ_min)
    λ_upper = min(λ_max_adhoc, λ_max)

    # initialization
    iter = 0
    crit = Inf
    λ_optimal = 0.0

    # start looping
    while crit > tol && iter < iter_max

        # update the multiplier
        λ_optimal = (λ_lower + λ_upper)/2

        # compute the associated results
        parameters_λ_optimal = para_func(η = η, a_min = a_min, a_size_neg = a_size_neg, λ = λ_optimal)
        variables_λ_optimal = var_func(parameters_λ_optimal, load_initial_values = 0)
        ED_λ_optimal = solve_func!(variables_λ_optimal, parameters_λ_optimal)

        # update search region
        if ED_λ_optimal > 0.0
            λ_upper = λ_optimal
        else
            λ_lower = λ_optimal
        end

        # check convergence
        crit = abs(ED_λ_optimal)

        # update the iteration number
        iter += 1

    end

    # re-run the results with the optimal multiplier
    parameters_λ_optimal = para_func(η = η, a_min = a_min, a_size_neg = a_size_neg, λ = λ_optimal)
    variables_λ_optimal = var_func(parameters_λ_optimal, load_initial_values = 0)
    ED_λ_optimal = solve_func!(variables_λ_optimal, parameters_λ_optimal)

    # return associated results
    return parameters_λ_min, variables_λ_min, parameters_λ_optimal, variables_λ_optimal
end

#=
parameters_optimal = para_func(η = results[11,1], λ = results[11,3], a_min = -3.50, a_size_neg = 701)
variables_optimal =  var_func(parameters_optimal)
solve_func!(variables_optimal, parameters_optimal)
plot(parameters_optimal.a_grid_neg, variables_optimal.q, seriestype=:scatter, legend=:none)
plot(parameters_optimal.a_grid_neg, -parameters_optimal.a_grid_neg.*variables_optimal.q, seriestype=:scatter, legend=:none)
=#

#=
println("Solving the model with $(Threads.nthreads()) threads in Julia...")
λ_optimal, parameters_optimal, variables_optimal = λ_optimal_func(parameters.η, parameters.a_grid[1])

data_spec = Any[#= 1=# "Number of Endowment"                parameters_optimal.e_size;
                #= 2=# "Number of Assets"                   parameters_optimal.a_size;
                #= 3=# "Number of Negative Assets"          parameters_optimal.a_size_neg;
                #= 4=# "Number of Positive Assets"          parameters_optimal.a_size_pos;
                #= 5=# "Number of Assets (for Density)"     parameters_optimal.a_size_μ;
                #= 6=# "Minimum of Assets"                  parameters_optimal.a_grid[1];
                #= 7=# "Maximum of Assets"                  parameters_optimal.a_grid[end];
                #= 8=# "Scale of Impatience"                parameters_optimal.ν_grid[1];
                #= 9=# "Probability of being Impatient"     parameters_optimal.ν_Γ[1,1];
                #=10=# "Exogenous Risk-free Rate"           parameters_optimal.i;
                #=11=# "Multiplier of Incentive Constraint" parameters_optimal.λ;
                #=12=# "Marginal Benifit of Net Worth"      parameters_optimal.γ;
                #=13=# "Diverting Fraction"                 parameters_optimal.θ;
                #=14=# "Asset-to-Debt Ratio (Supply)"       parameters_optimal.AD;
                #=15=# "Additional Opportunity Cost"        parameters_optimal.r_lp;
                #=16=# "Capital"                            parameters_optimal.K;
                #=17=# "Total Loans"                        variables_optimal.aggregate_var[1];
                #=18=# "Total Deposits"                     variables_optimal.aggregate_var[2];
                #=19=# "Net Worth"                          variables_optimal.aggregate_var[3];
                #=20=# "Asset-to-Debt Ratio (Demand)"       variables_optimal.aggregate_var[4]]

hl_LR = Highlighter(f      = (data,i,j) -> i == 14 || i == 20,
                    crayon = Crayon(background = :light_blue))

pretty_table(data_spec, ["Name", "Value"];
             alignment=[:l,:r],
             formatters = ft_round(4),
             body_hlines = [7,9,15],
             highlighters = hl_LR)
=#

# with financial frictions
η_grid = collect(0.80:-0.025:0.25)
η_size = length(η_grid)
results_NFF = zeros(η_size,13)
results_FF = zeros(η_size,13)

for η_i in 1:η_size
    # compute the optimal multipliers with different η
    if η_i == 1
        parameters_NFF, variables_NFF, parameters_FF, variables_FF = λ_optimal_func(η_grid[η_i], -3.50)
    else
        parameters_NFF, variables_NFF, parameters_FF, variables_FF = λ_optimal_func(η_grid[η_i], -3.50, λ_min_adhoc = results_FF[η_i-1,3])
    end

    # record results
    results_NFF[η_i,1] = parameters_NFF.η
    results_NFF[η_i,2] = parameters_NFF.i
    results_NFF[η_i,3] = parameters_NFF.λ
    results_NFF[η_i,4] = parameters_NFF.r_lp
    results_NFF[η_i,5] = parameters_NFF.K
    results_NFF[η_i,6:end] .= variables_NFF.aggregate_var

    results_FF[η_i,1] = parameters_FF.η
    results_FF[η_i,2] = parameters_FF.i
    results_FF[η_i,3] = parameters_FF.λ
    results_FF[η_i,4] = parameters_FF.r_lp
    results_FF[η_i,5] = parameters_FF.K
    results_FF[η_i,6:end] .= variables_FF.aggregate_var
end

header = ["η", "i", "λ", "lp", "K", "B", "D", "N", "(K+B)/D", "% of Filers", "% in Debt", "Debt-to_Income", "Avg Loan Rate"]
pretty_table(results_NFF, header, formatters = ft_round(8))
pretty_table(results_FF, header, formatters = ft_round(8))
@save "06012021_results_eta_0.25_0.80.bson" results_NFF results_FF header

#=
plot(results[:,1], results[:,3], seriestype=:scatter, legend=:none, title="Multiplier")
plot(results[:,1], results[:,4], seriestype=:scatter, legend=:none, title="Liquidity Premium")
plot(results[:,1], results[:,9], seriestype=:scatter, legend=:none, title="Leverage")
plot(results[:,1], results[:,10]*100, seriestype=:scatter, legend=:none, title="Percentage of Filers")


function _func(
    η_results::Array{Float64,2}
    )

    CEV_V_results = zeros(a_size, e_size, ν_size, η_size)
    CEV_μ_results = zeros(a_size, e_size, ν_size, η_size)

    for η_i in 1:η_size

        parameters_η = para_func(η = η_grid[η_i], a_min = a_min, a_size_neg = a_size_neg, λ = λ_optimal)
        variables_η = var_func(parameters_η)

        CEV_V_results[:,:,:,η_i] .= variables_η.V
        CEV_μ_results[:,:,:,η_i] .= variables_η.μ
    end
end
=#
