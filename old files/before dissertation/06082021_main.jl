#===========================#
# Import packages and files #
#===========================#
using Dierckx
using FLOWMath
using Distributions
using JLD2: @save, @load
using LinearAlgebra: norm
# using Measures
using Optim
using Parameters: @unpack
using Plots
using PrettyTables
using ProgressMeter
using QuantEcon: gridmake, rouwenhorst, tauchen, stationary_distributions
using Roots
using UnicodePlots
using Interpolations

# print out the number of threads
println("Julia is running with $(Threads.nthreads()) threads...")

#==================#
# Define functions #
#==================#
function parameters_function(;
    β::Real = 0.94,                 # discount factor (households)
    β_f::Real = 0.96,               # discount factor (bank)
    r_f::Real = 1.0 / β_f - 1.0,    # risk-free rate
    τ::Real = 0.04,                 # transaction cost
    σ::Real = 2.00,                 # CRRA coefficient
    η::Real = 0.355,                # garnishment rate
    δ::Real = 0.08,                 # depreciation rate
    α::Real = 1.0 / 3.0,            # capital share
    ψ::Real = 0.972,                # exogenous retention ratio
    λ::Real = 0.00,                 # multiplier of incentive constraint
    θ::Real = 0.381,                # diverting fraction
    e_ρ::Real = 0.9630,             # AR(1) of persistent endowment shock
    e_σ::Real = 0.1300,             # s.d. of persistent endowment shock
    e_size::Integer = 9,            # number of persistent endowment shock
    t_σ::Real = 0.35,               # s.d. of transitory endowment shock
    t_size::Integer = 3,            # number of transitory endowment shock
    ν_s::Real = 0.00,               # scale of patience
    ν_p::Real = 0.00,               # probability of patience
    ν_size::Integer = 2,            # number of preference shock
    a_min::Real = -8.0,             # min of asset holding
    a_max::Real = 300.0,            # max of asset holding
    a_size_neg::Integer = 401,      # number of grid of negative asset holding for VFI
    a_size_pos::Integer = 51,       # number of grid of positive asset holding for VFI
    a_degree::Integer = 3,          # curvature of the positive asset gridpoints
    a_size_pos_μ::Integer = 751,    # number of grid of positive asset holding for distribution
    )
    """
    contruct an immutable object containg all paramters
    """

    # persistent endowment shock
    e_MC = tauchen(e_size, e_ρ, e_σ, 0.0, 3)
    e_Γ = e_MC.p
    e_grid = collect(e_MC.state_values)
    e_SD = stationary_distributions(e_MC)[]
    e_SS = sum(e_SD .* e_grid)

    # transitory endowment shock
    t_grid = quantile.(Normal(0.0, t_σ), collect(range(0.0, 1.0; step = 1.0 / (t_size + 1))))
    t_grid = t_grid[2:(end-1)]
    t_Γ = repeat([1.0 / t_size], inner = t_size)
    t_SS = sum(t_Γ .* t_grid)

    # preference schock
    ν_grid = [ν_s, 1.0]
    ν_Γ = [ν_p, 1.0 - ν_p]

    # idiosyncratic state
    x_grid = gridmake(e_grid, t_grid, ν_grid)
    x_ind = gridmake(1:e_size, 1:t_size, 1:ν_size)
    x_size = e_size * t_size * ν_size

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length = a_size_neg))
    a_grid_pos = ((range(0.0, stop = a_size_pos - 1, length = a_size_pos) / (a_size_pos - 1)) .^ a_degree) * a_max
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero, a_grid)[]

    # asset holding grid for μ
    a_size_neg_μ = a_size_neg
    a_grid_neg_μ = collect(range(a_min, 0.0, length = a_size_neg_μ))
    a_grid_pos_μ = collect(range(0.0, a_max, length = a_size_pos_μ))
    a_grid_μ = cat(a_grid_neg_μ[1:(end-1)], a_grid_pos_μ, dims = 1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(iszero, a_grid_μ)[]

    # compute equilibrium prices and quantities
    ξ = (1.0 - ψ) / (1 - λ - ψ)
    Λ = β_f * (1.0 - ψ + ψ * ξ)
    LR = ξ / θ
    KL_to_D_ratio = LR / (LR - 1.0)
    ι = λ * θ / Λ
    r_k = r_f + ι
    E = exp(e_SS + t_SS)
    K = E * ((r_k + δ) / α)^(1.0 / (α - 1.0))
    w = (1.0 - α) * (K / E)^α

    # return values
    return (
        β = β,
        β_f = β_f,
        r_f = r_f,
        τ = τ,
        σ = σ,
        η = η,
        δ = δ,
        α = α,
        ψ = ψ,
        λ = λ,
        θ = θ,
        a_degree = a_degree,
        e_ρ = e_ρ,
        e_σ = e_σ,
        e_size = e_size,
        e_Γ = e_Γ,
        e_grid = e_grid,
        t_σ = t_σ,
        t_size = t_size,
        t_Γ = t_Γ,
        t_grid = t_grid,
        ν_s = ν_s,
        ν_p = ν_p,
        ν_size = ν_size,
        ν_Γ = ν_Γ,
        ν_grid = ν_grid,
        x_grid = x_grid,
        x_ind = x_ind,
        x_size = x_size,
        a_grid = a_grid,
        a_grid_neg = a_grid_neg,
        a_grid_pos = a_grid_pos,
        a_size = a_size,
        a_size_neg = a_size_neg,
        a_size_pos = a_size_pos,
        a_ind_zero = a_ind_zero,
        a_grid_μ = a_grid_μ,
        a_grid_neg_μ = a_grid_neg_μ,
        a_grid_pos_μ = a_grid_pos_μ,
        a_size_μ = a_size_μ,
        a_size_neg_μ = a_size_neg_μ,
        a_size_pos_μ = a_size_pos_μ,
        a_ind_zero_μ = a_ind_zero_μ,
        ξ = ξ,
        Λ = Λ,
        LR = LR,
        KL_to_D_ratio = KL_to_D_ratio,
        ι = ι,
        r_k = r_k,
        E = E,
        K = K,
        w = w,
    )
end

mutable struct MutableAggregateVariables
    """
    construct a type for mutable aggregate variables
    """
    L::Real
    D::Real
    N::Real
    leverage_ratio::Real
    KL_to_D_ratio::Real
    debt_to_earning_ratio::Real
    share_of_filers::Real
    share_of_involuntary_filers::Real
    share_in_debts::Real
    avg_loan_rate::Real
    avg_loan_rate_pw::Real
end

mutable struct MutableVariables
    """
    construct a type for mutable variables
    """
    R::Array{Float64,2}
    q::Array{Float64,2}
    rbl::Array{Float64,2}
    V::Array{Float64,4}
    V_d::Array{Float64,3}
    V_nd::Array{Float64,4}
    policy_a::Array{Float64,4}
    threshold_a::Array{Float64,3}
    threshold_e::Array{Float64,3}
    μ::Array{Float64,4}
    aggregate_variables::MutableAggregateVariables
end

function min_bounds_function(obj::Function, grid_min::Real, grid_max::Real; grid_length::Integer = 50, obj_range::Integer = 1)
    """
    compute bounds for minimization
    """

    grid = range(grid_min, grid_max, length = grid_length)
    grid_size = length(grid)
    obj_grid = obj.(grid)
    obj_index = argmin(obj_grid)
    if obj_index < (1 + obj_range)
        lb = grid_min
        @inbounds ub = grid[obj_index+obj_range]
    elseif obj_index > (grid_size - obj_range)
        @inbounds lb = grid[obj_index-obj_range]
        ub = grid_max
    else
        @inbounds lb = grid[obj_index-obj_range]
        @inbounds ub = grid[obj_index+obj_range]
    end
    return lb, ub
end

function zero_bounds_function(V_d::Real, V_nd::Array{Float64,1}, a_grid::Array{Float64,1})
    """
    compute bounds for (zero) root finding
    """

    @inbounds lb = a_grid[minimum(findall(V_nd .> V_d))]
    @inbounds ub = a_grid[maximum(findall(V_nd .< V_d))]
    return lb, ub
end

function utility_function(c::Real, γ::Real)
    """
    compute utility of CRRA utility function with coefficient γ
    """

    if c > 0.0
        return γ == 1.0 ? log(c) : 1.0 / ((1.0 - γ) * c^(γ - 1.0))
    else
        return -Inf
    end
end

function log_function(threshold_e::Real)
    """
    adjusted log funciton where assigning -Inf to negative domain
    """

    if threshold_e > 0.0
        return log(threshold_e)
    else
        return -Inf
    end
end

function repayment_function(e_i::Integer, a_p::Real, threshold::Real, parameters::NamedTuple; wage_garnishment::Bool = true)
    """
    evaluate repayment analytically with and without wage garnishment for a given defaulting threshold in e'
    """

    # unpack parameters
    @unpack e_grid, e_ρ, e_σ, η = parameters

    # compute expected repayment amount
    @inbounds e_μ = e_ρ * e_grid[e_i]

    # (1) not default
    default_prob = cdf(Normal(e_μ, e_σ), threshold)
    amount_repay = -a_p * (1.0 - default_prob)

    # (2) default and reclaiming wage garnishment is enabled
    amount_default = 0.0
    if wage_garnishment == true
        default_adjusted_prob = cdf(Normal(e_μ + e_σ^2.0, e_σ), threshold)
        amount_default = η * exp(e_μ + e_σ^2.0 / 2.0) * default_adjusted_prob
    end

    return total_amount = amount_repay + amount_default
end

function variables_function(parameters::NamedTuple)
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack a_size, a_grid, a_size_neg, a_grid_neg, a_size_μ, e_size, e_grid, t_size, t_grid, t_Γ, e_ρ, e_σ, ν_size, w, r_f, τ, ι, η, σ = parameters

    # define repayment probability, pricing function, and risky borrowing limit
    R = zeros(a_size_neg, e_size)
    q = ones(a_size, e_size) ./ (1.0 + r_f)
    rbl = zeros(e_size, 2)
    for e_i = 1:e_size
        for a_p_i = 1:(a_size_neg-1)
            @inbounds a_p = a_grid_neg[a_p_i]
            for t_p_i = 1:t_size
                @inbounds threshold = log_function(-a_p / w) - t_grid[t_p_i]
                @inbounds R[a_p_i, e_i] += t_Γ[t_p_i] * repayment_function(e_i, a_p, threshold, parameters)
            end
            @inbounds q[a_p_i, e_i] = R[a_p_i, e_i] / ((-a_p) * (1.0 + r_f + τ + ι))
        end

        qa_funcion_itp = Akima(a_grid, q[:, e_i] .* a_grid)
        qa_funcion(a_p) = qa_funcion_itp(a_p)
        @inbounds rbl_lb, rbl_ub = min_bounds_function(qa_funcion, a_grid[1], 0.0)
        res_rbl = optimize(qa_funcion, rbl_lb, rbl_ub)
        @inbounds rbl[e_i, 1] = Optim.minimizer(res_rbl)
        @inbounds rbl[e_i, 2] = Optim.minimum(res_rbl)
    end

    # define value and policy functions
    V = zeros(a_size, e_size, t_size, ν_size)
    V_d = zeros(e_size, t_size, ν_size)
    V_nd = zeros(a_size, e_size, t_size, ν_size)
    policy_a = zeros(a_size, e_size, t_size, ν_size)

    # define thresholds conditional on endowment or asset
    threshold_a = zeros(e_size, t_size, ν_size)
    threshold_e = zeros(a_size, t_size, ν_size)

    # define cross-sectional distribution
    μ_size = a_size_μ * e_size * t_size * ν_size
    μ = ones(a_size_μ, e_size, t_size, ν_size) ./ μ_size

    # define aggregate variables
    L = 0.0
    D = 0.0
    N = 0.0
    leverage_ratio = 0.0
    KL_to_D_ratio = 0.0
    debt_to_earning_ratio = 0.0
    share_of_filers = 0.0
    share_of_involuntary_filers = 0.0
    share_in_debts = 0.0
    avg_loan_rate = 0.0
    avg_loan_rate_pw = 0.0
    aggregate_variables =
        MutableAggregateVariables(L, D, N, leverage_ratio, KL_to_D_ratio, debt_to_earning_ratio, share_of_filers, share_of_involuntary_filers, share_in_debts, avg_loan_rate, avg_loan_rate_pw)

    # return outputs
    variables = MutableVariables(R, q, rbl, V, V_d, V_nd, policy_a, threshold_a, threshold_e, μ, aggregate_variables)
    return variables
end

function EV_itp_function(a_p::Real, e_i::Integer, V_d_p::Array{Float64,3}, V_nd_p::Array{Float64,4}, threshold_a::Array{Float64,3}, parameters::NamedTuple)
    """
    construct interpolated expected value function
    """

    # unpack parameters
    @unpack a_grid, e_size, e_Γ, t_size, t_Γ, ν_size, ν_Γ = parameters

    # construct container
    EV = 0.0

    # loop nested functions
    for e_p_i = 1:e_size, t_p_i = 1:t_size, ν_p_i = 1:ν_size

        # interpolated non-defaulting value function
        @inbounds @views V_nd_p_Non_Inf = findall(V_nd_p[:, e_p_i, t_p_i, ν_p_i] .!= -Inf)
        @inbounds @views a_grid_itp = a_grid[V_nd_p_Non_Inf]
        @inbounds @views V_nd_p_grid_itp = V_nd_p[V_nd_p_Non_Inf, e_p_i, t_p_i, ν_p_i]
        V_nd_p_itp = Akima(a_grid_itp, V_nd_p_grid_itp)

        # interpolated value function based on defaulting threshold
        @inbounds V_p_itp(a_p) = a_p >= threshold_a[e_p_i, t_p_i, ν_p_i] ? V_nd_p_itp(a_p) : V_d_p[e_p_i, t_p_i, ν_p_i]

        # update expected value
        @inbounds EV += ν_Γ[ν_p_i] * t_Γ[t_p_i] * e_Γ[e_i, e_p_i] * V_p_itp(a_p)
    end

    # return value
    return EV
end

function EV_function(e_i::Integer, V_nd_p::Array{Float64,4}, parameters::NamedTuple)
    """
    construct expected non-defaulting value function
    """

    # unpack parameters
    @unpack a_size, e_size, e_Γ, t_size, t_Γ, ν_size, ν_Γ = parameters

    # construct container
    EV = zeros(a_size)

    # update expected value
    for e_p_i = 1:e_size, t_p_i = 1:t_size, ν_p_i = 1:ν_size
        @inbounds @views EV += ν_Γ[ν_p_i] * t_Γ[t_p_i] * e_Γ[e_i, e_p_i] * V_nd_p[:, e_p_i, t_p_i, ν_p_i]
    end

    # repalce NaN with -Inf
    replace!(EV, NaN => -Inf)

    # return value
    return EV
end

function EV_function(e_i::Integer, V_d_p::Array{Float64,3}, parameters::NamedTuple)
    """
    construct expected defaulting value
    """

    # unpack parameters
    @unpack a_size, e_size, e_Γ, t_size, t_Γ, ν_size, ν_Γ = parameters

    # construct container
    EV = 0.0

    # update expected value
    for e_p_i = 1:e_size, t_p_i = 1:t_size, ν_p_i = 1:ν_size
        @inbounds @views EV += ν_Γ[ν_p_i] * t_Γ[t_p_i] * e_Γ[e_i, e_p_i] * V_d_p[e_p_i, t_p_i, ν_p_i]
    end

    # return value
    return EV
end

function value_and_policy_function!(V_p::Array{Float64,4}, V_d_p::Array{Float64,3}, V_nd_p::Array{Float64,4}, variables::MutableVariables, parameters::NamedTuple; slow_updating::Real = 1.0)
    """
    update value and policy functions
    """

    # unpack parameters
    @unpack a_size, a_grid = parameters
    @unpack e_size, e_grid = parameters
    @unpack t_size, t_grid = parameters
    @unpack ν_size, ν_grid = parameters
    @unpack β, σ, η, w, r_f, ι = parameters

    # loop over all states
    for e_i = 1:e_size, t_i = 1:t_size, ν_i = 1:ν_size

        # construct earning
        @inbounds y = w * exp(e_grid[e_i] + t_grid[t_i])

        # extract risky borrowing limit and maximum discounted borrowing amount
        @inbounds @views rbl_a, rbl_qa = variables.rbl[e_i, :]

        # construct interpolated discounted borrowing amount functions
        @inbounds @views qa = variables.q[:, e_i] .* a_grid
        qa_function_itp = Akima(a_grid, qa)

        # extract preference
        @inbounds ν = ν_grid[ν_i]

        # compute the next-period discounted expected value funtions and interpolated functions
        # V_hat_itp(a_p) = ν * β * EV_itp_function(a_p, e_i, V_d_p, V_nd_p, variables.threshold_a, parameters)
        V_hat = ν * β * EV_function(e_i, V_p, parameters)
        # V_hat_itp = Akima(a_grid, V_hat)
        V_hat_itp = linear_interpolation(a_grid, V_hat, extrapolation_bc=Line()) 

        #=
        EV_nd_hat = EV_function(e_i, V_nd_p, parameters)
        EV_d_hat = EV_function(e_i, V_d_p, parameters)
        EV_nd_hat_Non_Inf = findall(EV_nd_hat .!= -Inf)
        EV_nd_hat_itp = Akima(a_grid[EV_nd_hat_Non_Inf], EV_nd_hat[EV_nd_hat_Non_Inf])
        EV_hat_diff_itp(a_p) = EV_nd_hat_itp(a_p) - EV_d_hat
        threshold_a_p = -Inf
        if minimum(EV_nd_hat_Non_Inf) < EV_d_hat
            EV_hat_diff_lb, EV_hat_diff_ub = zero_bounds_function(EV_d_hat, EV_nd_hat, a_grid)
            threshold_a_p = find_zero(a -> EV_hat_diff_itp(a), (EV_hat_diff_lb, EV_hat_diff_ub), Bisection())
        end
        EV_hat_itp(a_p) = a_p >= threshold_a_p ? EV_nd_hat_itp(a_p) : EV_d_hat
        V_hat_itp(a_p) = ν * β * EV_hat_itp(a_p)
        =#

        # compute defaulting value
        @inbounds variables.V_d[e_i, t_i, ν_i] = utility_function((1 - η) * y, σ) + V_hat_itp(0.0)

        # initialize policy function
        @inbounds @views variables.policy_a[:, e_i, t_i, ν_i] .= -Inf

        # compute non-defaulting value
        Threads.@threads for a_i = 1:a_size

            # cash on hand
            @inbounds CoH = y + a_grid[a_i]

            if (CoH - rbl_qa) >= 0.0

                # define optimization problem
                object_nd(a_p) = -(utility_function(CoH - qa_function_itp(a_p), σ) + V_hat_itp(a_p))

                #=
                # monotonicity of policy function and if used Threads.@threads must be removed from the loop over current wealth
                if a_i > 1
                    if variables.policy_a[a_i-1, e_i, t_i, ν_i] != -Inf
                        lb = variables.policy_a[a_i-1, e_i, t_i, ν_i] - eps()
                        ub = CoH * (1 + r_f + ι)
                    else
                        lb = rbl_a - eps()
                        ub = CoH * (1 + r_f + ι)
                    end
                else
                    lb = rbl_a - eps()
                    ub = CoH * (1 + r_f + ι)
                end
                =#

                lb, ub = min_bounds_function(object_nd, rbl_a - eps(), CoH)
                res_nd = optimize(object_nd, lb, ub)
                @inbounds variables.V_nd[a_i, e_i, t_i, ν_i] = -Optim.minimum(res_nd)
                @inbounds variables.policy_a[a_i, e_i, t_i, ν_i] = Optim.minimizer(res_nd)

                if variables.V_nd[a_i, e_i, t_i, ν_i] > variables.V_d[e_i, t_i, ν_i]
                    # repayment
                    @inbounds variables.V[a_i, e_i, t_i, ν_i] = variables.V_nd[a_i, e_i, t_i, ν_i]
                else
                    # voluntary default
                    @inbounds variables.V[a_i, e_i, t_i, ν_i] = variables.V_d[e_i, t_i, ν_i]
                end
            else
                # involuntary default
                @inbounds variables.V_nd[a_i, e_i, t_i, ν_i] = utility_function(0.0, σ)
                @inbounds variables.V[a_i, e_i, t_i, ν_i] = variables.V_d[e_i, t_i, ν_i]
            end
        end
    end

    # slow updating
    if slow_updating != 1.0
        variables.V = slow_updating * variables.V + (1.0 - slow_updating) * V_p
        variables.V_d = slow_updating * variables.V_d + (1.0 - slow_updating) * V_d_p
        variables.V_nd = slow_updating * variables.V_nd + (1.0 - slow_updating) * V_nd_p
    end
end

function threshold_function!(variables::MutableVariables, parameters::NamedTuple)
    """
    update thresholds
    """

    # unpack parameters
    @unpack ν_size, t_size, t_grid, e_size, e_grid, a_size, a_grid, w = parameters

    for ν_i = 1:ν_size, t_i = 1:t_size

        # defaulting thresholds in wealth (a)
        for e_i = 1:e_size
            @inbounds @views V_nd_Non_Inf = findall(variables.V_nd[:, e_i, t_i, ν_i] .!= -Inf)
            @inbounds @views a_grid_itp = a_grid[V_nd_Non_Inf]
            @inbounds @views V_nd_grid_itp = variables.V_nd[V_nd_Non_Inf, e_i, t_i, ν_i]
            # V_nd_itp = Akima(a_grid_itp, V_nd_grid_itp)
            V_nd_itp = linear_interpolation(a_grid_itp, V_nd_grid_itp, extrapolation_bc=Line()) 
            @inbounds V_diff_itp(a) = V_nd_itp(a) - variables.V_d[e_i, t_i, ν_i]
            if minimum(V_nd_grid_itp) > variables.V_d[e_i, t_i, ν_i]
                @inbounds variables.threshold_a[e_i, t_i, ν_i] = -Inf
            else
                @inbounds V_diff_lb, V_diff_ub = zero_bounds_function(variables.V_d[e_i, t_i, ν_i], variables.V_nd[:, e_i, t_i, ν_i], a_grid)
                @inbounds variables.threshold_a[e_i, t_i, ν_i] = find_zero(a -> V_diff_itp(a), (V_diff_lb, V_diff_ub), Bisection())
            end
        end

        # defaulting thresholds in endowment (e)
        @inbounds @views thres_a_Non_Inf = findall(variables.threshold_a[:, t_i, ν_i] .!= -Inf)
        @inbounds @views thres_a_grid_itp = -variables.threshold_a[thres_a_Non_Inf, t_i, ν_i]
        earning_grid_itp = w * exp.(e_grid[thres_a_Non_Inf] .+ t_grid[t_i])
        # threshold_earning_itp = Spline1D(thres_a_grid_itp, earning_grid_itp; k = 1, bc = "extrapolate")
        threshold_earning_itp = linear_interpolation(thres_a_grid_itp, earning_grid_itp, extrapolation_bc=Line())
        Threads.@threads for a_i = 1:a_size
            @inbounds earning_thres = threshold_earning_itp(-a_grid[a_i])
            e_thres = earning_thres > 0.0 ? log(earning_thres / w) - t_grid[t_i] : -Inf
            @inbounds variables.threshold_e[a_i, t_i, ν_i] = e_thres
        end
    end
end

function pricing_and_rbl_function!(variables::MutableVariables, parameters::NamedTuple)
    """
    update pricing function and borrowing risky limit
    """

    # unpack parameters
    @unpack r_f, τ, ι, a_size_neg, a_grid, e_size, e_grid, t_size, t_Γ, ν_size, ν_Γ = parameters

    # initialization
    variables.R .= 0.0
    variables.q .= 1.0 / (1.0 + r_f)

    # loop over states
    for e_i = 1:e_size

        # repayment probability and pricing funciton
        Threads.@threads for a_p_i = 1:(a_size_neg-1)
            @inbounds a_p = a_grid[a_p_i]
            for t_p_i = 1:t_size, ν_p_i = 1:ν_size
                @inbounds variables.R[a_p_i, e_i] += ν_Γ[ν_p_i] * t_Γ[t_p_i] * repayment_function(e_i, a_p, variables.threshold_e[a_p_i, t_p_i, ν_p_i], parameters)
            end
            @inbounds variables.q[a_p_i, e_i] = variables.R[a_p_i, e_i] / ((-a_p) * (1.0 + r_f + τ + ι))
        end

        # risky borrowing limit and maximum discounted borrwoing amount
        @inbounds @views q_e = variables.q[:, e_i]
        qa_function_itp = Akima(a_grid, q_e .* a_grid)
        qa_function(x) = qa_function_itp(x)
        rbl_lb, rbl_ub = min_bounds_function(qa_function, a_grid[1], 0.0)
        res_rbl = optimize(qa_function, rbl_lb, rbl_ub)
        @inbounds variables.rbl[e_i, 1] = Optim.minimizer(res_rbl)
        @inbounds variables.rbl[e_i, 2] = Optim.minimum(res_rbl)
    end
end

function solve_value_and_pricing_function!(variables::MutableVariables, parameters::NamedTuple; tol::Real = 1E-8, iter_max::Integer = 1000, figure_track::Bool = false, slow_updating::Real = 1.0)
    """
    solve household and banking problems using one-loop algorithm
    """

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    # prog = ProgressThresh(tol, "Solving household and banking problems (one-loop): ")

    # construct containers
    V_p = similar(variables.V)
    V_d_p = similar(variables.V_d)
    V_nd_p = similar(variables.V_nd)
    q_p = similar(variables.q)

    while crit > tol && iter < iter_max

        # copy previous values
        copyto!(V_p, variables.V)
        copyto!(V_d_p, variables.V_d)
        copyto!(V_nd_p, variables.V_nd)
        copyto!(q_p, variables.q)

        # value and policy functions
        value_and_policy_function!(V_p, V_d_p, V_nd_p, variables, parameters; slow_updating = slow_updating)

        # thresholds
        threshold_function!(variables, parameters)

        # pricing function and borrowing risky limit
        pricing_and_rbl_function!(variables, parameters)

        # check convergence
        V_crit = norm(variables.V .- V_p, Inf)
        q_crit = norm(variables.q .- q_p, Inf)
        crit = max(V_crit, q_crit)

        #=
        V_crit_index = findall(abs.(variables.V .- V_p) .== V_crit)
        println("")
        println("V_crit = $V_crit")
        println("V_crit_max happends at $V_crit_index")
        println("q_crit = $q_crit")
        =#

        # report progress
        # ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1

        # manually report convergence progress
        println("Solving household and banking problems (one-loop): iter = $iter and crit = $crit > tol = $tol")

        # tracking figures
        if figure_track == true

            # add new line
            println()

            # discounted bond price
            plt_q = lineplot(
                parameters.a_grid_neg,
                variables.q[1:parameters.a_size_neg, end],
                name = "e = $(round(parameters.e_grid[end],digits=2))",
                title = "discounted bond price",
                xlim = [round(parameters.a_grid[1], digits = 1), 0.0],
                ylim = [0.0, ceil(maximum(variables.q[1:parameters.a_size_neg, end]); digits = 1)],
                width = 50,
                height = 10,
            )
            for e_i = (parameters.e_size-1):(-1):1
                lineplot!(plt_q, parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, e_i], name = "e = $(round(parameters.e_grid[e_i],digits=2))")
            end
            println(plt_q)

            # discounted borrowing amount
            plt_qa = lineplot(
                parameters.a_grid_neg,
                -variables.q[1:parameters.a_size_neg, end] .* parameters.a_grid_neg,
                name = "e = $(round(parameters.e_grid[end],digits=2))",
                title = "discounted borrowing amount",
                xlim = [round(parameters.a_grid[1], digits = 1), 0.0],
                ylim = [0.0, ceil(maximum(-variables.q[1:parameters.a_size_neg, end] .* parameters.a_grid_neg); digits = 1)],
                width = 50,
                height = 10,
            )
            for e_i = (parameters.e_size-1):(-1):1
                lineplot!(plt_qa, parameters.a_grid_neg, -variables.q[1:parameters.a_size_neg, e_i] .* parameters.a_grid_neg, name = "e = $(round(parameters.e_grid[e_i],digits=2))")
            end
            println(plt_qa)
        end
    end
end

function stationary_distribution_function!(μ_p::Array{Float64,4}, variables::MutableVariables, parameters::NamedTuple)
    """
    update stationary distribution
    """

    # unpack parameters
    @unpack e_size, e_Γ, t_size, t_Γ, ν_size, ν_Γ, a_grid, a_size_μ, a_grid_μ, a_ind_zero_μ = parameters

    # initialization
    variables.μ .= 0.0

    for e_i = 1:e_size, t_i = 1:t_size, ν_i = 1:ν_size

        # interpolated decision rules
        @inbounds @views policy_a_Non_Inf = findall(variables.policy_a[:, e_i, t_i, ν_i] .!= -Inf)
        @inbounds policy_a_itp = Akima(a_grid[policy_a_Non_Inf], variables.policy_a[policy_a_Non_Inf, e_i, t_i, ν_i])
        @inbounds policy_d_itp(a_μ) = a_μ > variables.threshold_a[e_i, t_i, ν_i] ? 0.0 : 1.0

        # loop over the dimension of asset holding
        for a_μ_i = 1:a_size_μ

            # extract wealth and compute asset choice
            @inbounds a_μ = a_grid_μ[a_μ_i]
            @inbounds a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])

            # locate it on the original grid
            a_p_lb = findall(a_grid_μ .<= a_p)[end]
            a_p_ub = findall(a_p .<= a_grid_μ)[1]

            # compute weights
            if a_p_lb != a_p_ub
                @inbounds a_p_lower = a_grid_μ[a_p_lb]
                @inbounds a_p_upper = a_grid_μ[a_p_ub]
                weight_lower = (a_p_upper - a_p) / (a_p_upper - a_p_lower)
                weight_upper = (a_p - a_p_lower) / (a_p_upper - a_p_lower)
            else
                weight_lower = 0.5
                weight_upper = 0.5
            end

            # loop over the dimension of exogenous individual states
            for e_p_i = 1:e_size, t_p_i = 1:t_size, ν_p_i = 1:ν_size
                @inbounds variables.μ[a_p_lb, e_p_i, t_p_i, ν_p_i] += (1.0 - policy_d_itp(a_μ)) * ν_Γ[ν_p_i] * t_Γ[t_p_i] * e_Γ[e_i, e_p_i] * weight_lower * μ_p[a_μ_i, e_i, t_i, ν_i]
                @inbounds variables.μ[a_p_ub, e_p_i, t_p_i, ν_p_i] += (1.0 - policy_d_itp(a_μ)) * ν_Γ[ν_p_i] * t_Γ[t_p_i] * e_Γ[e_i, e_p_i] * weight_upper * μ_p[a_μ_i, e_i, t_i, ν_i]
                @inbounds variables.μ[a_ind_zero_μ, e_p_i, t_p_i, ν_p_i] += policy_d_itp(a_μ) * ν_Γ[ν_p_i] * t_Γ[t_p_i] * e_Γ[e_i, e_p_i] * μ_p[a_μ_i, e_i, t_i, ν_i]
            end
        end
    end
    variables.μ .= variables.μ ./ sum(variables.μ)
end

function solve_stationary_distribution_function!(variables::MutableVariables, parameters::NamedTuple; tol::Real = 1E-8, iter_max::Integer = 2000)
    """
    solve stationary distribution
    """

    # unpack parameters
    @unpack e_size, e_Γ, t_size, t_Γ, ν_size, ν_Γ, a_grid, a_size_μ, a_grid_μ, a_ind_zero_μ = parameters

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    # prog = ProgressThresh(tol, "Solving stationary distribution: ")

    # construct container
    μ_p = similar(variables.μ)

    while crit > tol && iter < iter_max

        # copy previous value
        copyto!(μ_p, variables.μ)

        # update stationary distribution
        stationary_distribution_function!(μ_p, variables, parameters)

        # check convergence
        crit = norm(variables.μ .- μ_p, Inf)

        # report preogress
        # ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1

        # manually report convergence progress
        println("Solving stationary distribution: iter = $iter and crit = $crit > tol = $tol")
    end
end

function solve_aggregate_variable_function!(variables::MutableVariables, parameters::NamedTuple)
    """
    compute equlibrium aggregate variables
    """

    # unpack parameters
    @unpack e_size, e_grid, t_size, t_grid, ν_size, a_grid, a_grid_neg, a_grid_pos = parameters
    @unpack a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ, a_grid_μ, a_size_μ = parameters
    @unpack K, w = parameters

    # initialize container
    variables.aggregate_variables.L = 0.0
    variables.aggregate_variables.D = 0.0
    variables.aggregate_variables.share_of_filers = 0.0
    variables.aggregate_variables.debt_to_earning_ratio = 0.0

    # construct auxiliary variables
    avg_loan_rate_num = 0.0
    avg_loan_rate_den = 0.0
    avg_loan_rate_pw_num = 0.0
    avg_loan_rate_pw_den = 0.0

    # total loans, deposits, share of filers, nad debt-to-earning ratio
    for e_i = 1:e_size, t_i = 1:t_size, ν_i = 1:ν_size

        # interpolated decision rules
        @inbounds @views policy_a_Non_Inf = findall(variables.policy_a[:, e_i, t_i, ν_i] .!= -Inf)
        @inbounds policy_a_itp = Akima(a_grid[policy_a_Non_Inf], variables.policy_a[policy_a_Non_Inf, e_i, t_i, ν_i])
        @inbounds policy_d_itp(a_μ) = a_μ > variables.threshold_a[e_i, t_i, ν_i] ? 0.0 : 1.0

        # interpolated discounted borrowing amount
        @inbounds @views q_e = variables.q[:, e_i]
        q_function_itp = Akima(a_grid, q_e)
        qa_function_itp = Akima(a_grid, q_e .* a_grid)

        # loop over the dimension of asset holding
        for a_μ_i = 1:a_size_μ

            # extract wealth and compute asset choice
            @inbounds a_μ = a_grid_μ[a_μ_i]
            @inbounds a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])

            if a_p < 0.0
                # total loans
                @inbounds variables.aggregate_variables.L += -(variables.μ[a_μ_i, e_i, t_i, ν_i] * (1.0 - policy_d_itp(a_μ)) * qa_function_itp(a_p))

                # average loan rate
                avg_loan_rate_num += variables.μ[a_μ_i, e_i, t_i, ν_i] * (1.0 - policy_d_itp(a_μ)) * (1.0 / q_function_itp(a_p) - 1.0)
                avg_loan_rate_den += variables.μ[a_μ_i, e_i, t_i, ν_i] * (1.0 - policy_d_itp(a_μ))

                # average loan rate (persons-weighted)
                avg_loan_rate_pw_num += (1.0 - policy_d_itp(a_μ)) * (1.0 / q_function_itp(a_p) - 1.0)
                avg_loan_rate_pw_den += 1
            else
                # total deposits
                @inbounds variables.aggregate_variables.D += (variables.μ[a_μ_i, e_i, t_i, ν_i] * (1.0 - policy_d_itp(a_μ)) * qa_function_itp(a_p))
            end

            if a_μ < 0.0
                # share of filers
                @inbounds variables.aggregate_variables.share_of_filers += (variables.μ[a_μ_i, e_i, t_i, ν_i] * policy_d_itp(a_μ))

                # share of involuntary filers
                if w * exp(e_grid[e_i] + t_grid[t_i]) + a_μ - variables.rbl[e_i, 2] < 0.0
                    @inbounds variables.aggregate_variables.share_of_involuntary_filers += (variables.μ[a_μ_i, e_i, t_i, ν_i] * policy_d_itp(a_μ))
                end

                # debt-to-earning ratio
                @inbounds variables.aggregate_variables.debt_to_earning_ratio += variables.μ[a_μ_i, e_i, t_i, ν_i] * (-a_μ / (w * exp(e_grid[e_i] + t_grid[t_i])))
            end
        end
    end

    # average loan rate
    variables.aggregate_variables.avg_loan_rate = avg_loan_rate_num / avg_loan_rate_den
    variables.aggregate_variables.avg_loan_rate_pw = avg_loan_rate_pw_num / avg_loan_rate_pw_den

    # net worth
    variables.aggregate_variables.N = (K + variables.aggregate_variables.L) - variables.aggregate_variables.D

    # leverage ratio
    variables.aggregate_variables.leverage_ratio = (K + variables.aggregate_variables.L) / variables.aggregate_variables.N

    # capital-loan-to-deposit ratio
    variables.aggregate_variables.KL_to_D_ratio = (K + variables.aggregate_variables.L) / variables.aggregate_variables.D

    # share in debt
    variables.aggregate_variables.share_in_debts = sum(variables.μ[1:(a_ind_zero_μ-1), :, :, :])
end

function solve_economy_function!(variables::MutableVariables, parameters::NamedTuple; tol_h::Real = 1E-6, tol_μ::Real = 1E-8)
    """
    solve the economy with given liquidity multiplier ι
    """

    # solve household and banking problems
    solve_value_and_pricing_function!(variables, parameters; tol = tol_h, iter_max = 500, figure_track = false, slow_updating = 1.0)

    # solve the cross-sectional distribution
    # solve_stationary_distribution_function!(variables, parameters; tol = tol_μ, iter_max = 1000)

    # compute aggregate variables
    # solve_aggregate_variable_function!(variables, parameters)

    # compute the difference between demand and supply sides
    ED = variables.aggregate_variables.KL_to_D_ratio - parameters.KL_to_D_ratio

    # printout results
    data_spec = Any[
        "Wage Garnishment Rate" parameters.η #=1=#
        "Liquidity Multiplier" parameters.λ #=2=#
        "Asset-to-Debt Ratio (Demand)" variables.aggregate_variables.KL_to_D_ratio #=3=#
        "Asset-to-Debt Ratio (Supply)" parameters.KL_to_D_ratio  #=4=#
        "Difference" ED #=5=#
    ]
    pretty_table(data_spec; header = ["Name", "Value"], alignment = [:l, :r], formatters = ft_round(8), body_hlines = [2, 4])

    # return excess demand
    return ED
end

function optimal_multiplier_function(η::Real; λ_min_adhoc::Real = -Inf, λ_max_adhoc::Real = Inf, tol::Real = 1E-6, iter_max::Real = 500)
    """
    solve for optimal liquidity multiplier
    """

    # check the case of λ_min = 0.0
    λ_min = 0.0
    parameters_λ_min = parameters_function(η = η, λ = λ_min)
    variables_λ_min = variables_function(parameters_λ_min)
    ED_λ_min = solve_economy_function!(variables_λ_min, parameters_λ_min)
    if ED_λ_min > 0.0
        return parameters_λ_min, variables_λ_min, parameters_λ_min, variables_λ_min
    end

    # check the case of λ_max = 1-ψ^(1/2)
    λ_max = 1.0 - sqrt(parameters_λ_min.ψ)
    parameters_λ_max = parameters_function(η = η, λ = λ_max)
    variables_λ_max = variables_function(parameters_λ_max)
    ED_λ_max = solve_economy_function!(variables_λ_max, parameters_λ_max)
    if ED_λ_max < 0.0
        return parameters_λ_min, variables_λ_min, parameters_λ_max, variables_λ_max # meaning solution doesn't exist!
    end

    # fing the optimal λ using bisection
    λ_lower = max(λ_min_adhoc, λ_min)
    λ_upper = min(λ_max_adhoc, λ_max)

    # initialization
    iter = 0
    crit = Inf
    λ_optimal = 0.0
    parameters_λ_optimal = []
    variables_λ_optimal = []

    # solve equlibrium multiplier by bisection
    while crit > tol && iter < iter_max

        # update the multiplier
        λ_optimal = (λ_lower + λ_upper) / 2

        # compute the associated results
        parameters_λ_optimal = parameters_function(η = η, λ = λ_optimal)
        variables_λ_optimal = variables_function(parameters_λ_optimal)
        ED_λ_optimal = solve_economy_function!(variables_λ_optimal, parameters_λ_optimal)

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

    # return results
    return parameters_λ_min, variables_λ_min, parameters_λ_optimal, variables_λ_optimal
end

function results_η_function(; η_min::Real, η_max::Real, η_step::Real)
    """
    compute stationary equilibrium with various η
    """

    # initialize η grid
    η_grid = collect(η_max:-η_step:η_min)
    η_size = length(η_grid)

    # initialize pparameters
    parameters = parameters_function()
    @unpack a_size, a_size_μ, e_size, t_size, ν_size = parameters

    # initialize variables that will be saved
    var_names = [
        "Wage Garnishment Rate", #=1=#
        "Rental Rate", #=2=#
        "Liquidity Multiplier", #=3=#
        "Liquidity Premium", #=4=#
        "Capital", #=5=#
        "Loans", #=6=#
        "Deposits", #=7=#
        "Net Worth", #=8=#
        "Leverage Ratio", #=9=#
        "Share of Filers", #=10=#
        "Sahre in Debt", #=11=#
        "Debt-to-Earning Ratio", #=12=#
        "Average Loan Rate", #=13=#
    ]
    var_size = length(var_names)

    # initialize containers
    results_A_NFF = zeros(var_size, η_size)
    results_V_NFF = zeros(a_size, e_size, t_size, ν_size, η_size)
    results_μ_NFF = zeros(a_size_μ, e_size, t_size, ν_size, η_size)
    results_A_FF = zeros(var_size, η_size)
    results_V_FF = zeros(a_size, e_size, t_size, ν_size, η_size)
    results_μ_FF = zeros(a_size_μ, e_size, t_size, ν_size, η_size)

    # compute the optimal multipliers with different η
    for η_i = 1:η_size

        # if η_i > 1 then use previouse to narrow down searching area
        if η_i == 1
            parameters_NFF, variables_NFF, parameters_FF, variables_FF = optimal_multiplier_function(η_grid[η_i])
        else
            parameters_NFF, variables_NFF, parameters_FF, variables_FF = optimal_multiplier_function(η_grid[η_i]; λ_min_adhoc = results_A_FF[3, η_i-1])
        end

        # save results
        results_A_NFF[1, η_i] = parameters_NFF.η
        results_A_NFF[2, η_i] = parameters_NFF.r_k
        results_A_NFF[3, η_i] = parameters_NFF.λ
        results_A_NFF[4, η_i] = parameters_NFF.ι
        results_A_NFF[5, η_i] = parameters_NFF.K
        results_A_NFF[6, η_i] = variables_NFF.aggregate_variables.L
        results_A_NFF[7, η_i] = variables_NFF.aggregate_variables.D
        results_A_NFF[8, η_i] = variables_NFF.aggregate_variables.N
        results_A_NFF[9, η_i] = variables_NFF.aggregate_variables.leverage_ratio
        results_A_NFF[10, η_i] = variables_NFF.aggregate_variables.share_of_filers
        results_A_NFF[11, η_i] = variables_NFF.aggregate_variables.share_in_debts
        results_A_NFF[12, η_i] = variables_NFF.aggregate_variables.debt_to_earning_ratio
        results_A_NFF[13, η_i] = variables_NFF.aggregate_variables.avg_loan_rate_pw
        results_V_NFF[:, :, :, :, η_i] = variables_NFF.V
        results_μ_NFF[:, :, :, :, η_i] = variables_NFF.μ

        results_A_FF[1, η_i] = parameters_FF.η
        results_A_FF[2, η_i] = parameters_FF.r_k
        results_A_FF[3, η_i] = parameters_FF.λ
        results_A_FF[4, η_i] = parameters_FF.ι
        results_A_FF[5, η_i] = parameters_FF.K
        results_A_FF[6, η_i] = variables_FF.aggregate_variables.L
        results_A_FF[7, η_i] = variables_FF.aggregate_variables.D
        results_A_FF[8, η_i] = variables_FF.aggregate_variables.N
        results_A_FF[9, η_i] = variables_FF.aggregate_variables.leverage_ratio
        results_A_FF[10, η_i] = variables_FF.aggregate_variables.share_of_filers
        results_A_FF[11, η_i] = variables_FF.aggregate_variables.share_in_debts
        results_A_FF[12, η_i] = variables_FF.aggregate_variables.debt_to_earning_ratio
        results_A_FF[13, η_i] = variables_FF.aggregate_variables.avg_loan_rate_pw
        results_V_FF[:, :, :, :, η_i] = variables_FF.V
        results_μ_FF[:, :, :, :, η_i] = variables_FF.μ
    end

    # return results
    return var_names, results_A_NFF, results_V_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_μ_FF
end

function results_CEV_function(results_V::Array{Float64,5})
    """
    compute consumption equivalent variation (CEV) with various η compared to the smallest η (most lenient policy)
    """

    # initialize pparameters
    parameters = parameters_function()
    @unpack a_grid, a_size, a_grid_μ, a_size_μ, e_size, t_size, ν_size, σ = parameters

    # initialize result matrix
    η_size = size(results_V, 5)
    results_CEV = zeros(a_size_μ, e_size, t_size, ν_size, η_size)

    # compute CEV for different η compared to the smallest η
    for η_i = 1:η_size, e_i = 1:e_size, t_i = 1:t_size, ν_i = 1:ν_size
        @inbounds @views V_itp_new = Akima(a_grid, results_V[:, e_i, t_i, ν_i, η_i])
        @inbounds @views V_itp_old = Akima(a_grid, results_V[:, e_i, t_i, ν_i, end])
        for a_μ_i = 1:a_size_μ
            @inbounds a_μ = a_grid_μ[a_μ_i]
            @inbounds results_CEV[a_μ_i, e_i, t_i, ν_i, η_i] = (V_itp_new(a_μ) / V_itp_old(a_μ))^(1 / (1 - σ)) - 1
        end
    end

    # return results
    return parameters, results_CEV
end

#=================#
# Solve the model #
#=================#
parameters = parameters_function(λ = 0.0034137294588653328)
variables = variables_function(parameters)
solve_economy_function!(variables, parameters)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_ind_zero,:])
plot(parameters.a_grid_neg[371:end], variables.V[371:parameters.a_ind_zero,5,:,2])
plot(parameters.a_grid_neg[371:end], variables.V_nd[371:parameters.a_ind_zero,5,:,2])
plot(parameters.e_grid, variables.threshold_a[:,:,2])
plot(parameters.a_grid_neg, variables.threshold_e[1:parameters.a_ind_zero,:,2])

# parameters_NFF = parameters_function(λ = 0.0)
# variables_NFF = variables_function(parameters_NFF)
# solve_economy_function!(variables_NFF, parameters_NFF)

# parameters_λ_min, variables_λ_min, parameters_λ_optimal, variables_λ_optimal = optimal_multiplier_function(parameters.η)
# λ_optimal = parameters_λ_optimal.λ

#======================================================#
# Solve the model with different bankruptcy strictness #
#======================================================#
# var_names, results_A_NFF, results_V_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_μ_FF = results_η_function(η_min = 0.20, η_max = 0.80, η_step = 0.10)
# @save "results_eta.jld2" var_names results_A_NFF results_V_NFF results_μ_NFF results_A_FF results_V_FF results_μ_FF
# @load "C:/Users/User/Desktop/results_eta.jld2" var_names results_A_NFF results_V_NFF results_μ_NFF results_A_FF results_V_FF results_μ_FF

#=============================#
# Solve transitional dynamics #
#=============================#
# periods = 80

# mutable struct MutableAggregateVariables_T
#     """
#     construct a type for mutable aggregate variables of periods T
#     """
#     L::Array{Float64,1}
#     D::Array{Float64,1}
#     N::Array{Float64,1}
#     leverage_ratio::Array{Float64,1}
#     KL_to_D_ratio::Array{Float64,1}
#     debt_to_earning_ratio::Array{Float64,1}
#     share_of_filers::Array{Float64,1}
#     share_of_involuntary_filers::Array{Float64,1}
#     share_in_debts::Array{Float64,1}
#     avg_loan_rate::Array{Float64,1}
#     avg_loan_rate_pw::Array{Float64,1}
# end

# mutable struct MutableVariables_T
#     """
#     construct a type for mutable variables of periods T
#     """
#     R::Array{Float64,3}
#     q::Array{Float64,3}
#     rbl::Array{Float64,3}
#     V::Array{Float64,5}
#     V_d::Array{Float64,4}
#     V_nd::Array{Float64,5}
#     policy_a::Array{Float64,5}
#     threshold_a::Array{Float64,4}
#     threshold_e::Array{Float64,4}
#     μ::Array{Float64,5}
#     aggregate_variables::MutableAggregateVariables_T
# end

# function variables_T_function(parameters::NamedTuple, T_size::Integer)
#     """
#     construct a mutable object containing endogenous variables of periods T
#     """

#     # unpack parameters
#     @unpack a_size, a_grid, a_size_neg, a_grid_neg, a_size_μ, e_size, e_grid, t_size, t_grid, t_Γ, e_ρ, e_σ, ν_size, w, r_f, τ, ι, η, σ = parameters

#     # define repayment probability, pricing function, and risky borrowing limit
#     R = zeros(a_size_neg, e_size, T_size)
#     q = ones(a_size, e_size, T_size) ./ (1.0 + r_f)
#     rbl = zeros(e_size, 2, T_size)

#     # define value and policy functions
#     V = zeros(a_size, e_size, t_size, ν_size, T_size)
#     V_d = zeros(e_size, t_size, ν_size, T_size)
#     V_nd = zeros(a_size, e_size, t_size, ν_size, T_size)
#     policy_a = zeros(a_size, e_size, t_size, ν_size, T_size)

#     # define thresholds conditional on endowment or asset
#     threshold_a = zeros(e_size, t_size, ν_size, T_size)
#     threshold_e = zeros(a_size, t_size, ν_size, T_size)

#     # define cross-sectional distribution
#     μ_size = a_size_μ * e_size * t_size * ν_size
#     μ = ones(a_size_μ, e_size, t_size, ν_size, T_size) ./ μ_size

#     # define aggregate variables
#     L = zeros(T_size)
#     D = zeros(T_size)
#     N = zeros(T_size)
#     leverage_ratio = zeros(T_size)
#     KL_to_D_ratio = zeros(T_size)
#     debt_to_earning_ratio = zeros(T_size)
#     share_of_filers = zeros(T_size)
#     share_of_involuntary_filers = zeros(T_size)
#     share_in_debts = zeros(T_size)
#     avg_loan_rate = zeros(T_size)
#     avg_loan_rate_pw = zeros(T_size)
#     aggregate_variables =
#         MutableAggregateVariables_T(L, D, N, leverage_ratio, KL_to_D_ratio, debt_to_earning_ratio, share_of_filers, share_of_involuntary_filers, share_in_debts, avg_loan_rate, avg_loan_rate_pw)

#     # return outputs
#     variables = MutableVariables_T(R, q, rbl, V, V_d, V_nd, policy_a, threshold_a, threshold_e, μ, aggregate_variables)
#     return variables
# end

# function transitional_dynamic_λ_function(
#     λ_old::Real = 0.0,
#     λ_new::Real = 0.0034137294588653328;
#     periods::Integer = periods
#     )

#     # old stationary equilibrium
#     parameters_old = parameters_function(λ = λ_old)
#     variables_old = variables_function(parameters_old)
#     solve_economy_function!(variables_old, parameters_old)

#     # new stationary equilibrium
#     parameters_new = parameters_function(λ = λ_new)
#     variables_new = variables_function(parameters_new)
#     solve_economy_function!(variables_new, parameters_new)

#     # conjecture of leverage ratio
#     LR_path = zeros(periods)
#     ι_path = zeros(periods)
#     r_k_path = zeros(periods)
#     w_path = zeros(periods)

# end
