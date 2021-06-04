#===========================#
# Import packages and files #
#===========================#
using Dierckx
using FLOWMath
using Distributions
using LinearAlgebra: norm
using Optim
using Parameters: @unpack
using Plots
using ProgressMeter
using QuantEcon: rouwenhorst, tauchen, stationary_distributions
using Roots

# print out the number of threads
println("Julia is running with $(Threads.nthreads()) threads...")

#==================#
# Define functions #
#==================#
function parameters_function(;
    β::Real = 0.96,                 # discount factor (households)
    β_f::Real = 0.96,               # discount factor (bank)
    r_f::Real = 1.00 / β_f - 1.00,  # risk-free rate
    σ::Real = 2.00,                 # CRRA coefficient
    η::Real = 0.40,                 # garnishment rate
    δ::Real = 0.08,                 # depreciation rate
    α::Real = 1.0 / 3.0,            # capital share
    ψ::Real = 0.90,                 # exogenous dividend rate
    λ::Real = 0.00,                 # multiplier of incentive constraint
    θ::Real = 0.40,                 # diverting fraction
    e_ρ::Real = 0.95,               # AR(1) of endowment shock
    e_σ::Real = 0.10,               # s.d. of endowment shock
    e_size::Integer = 3,            # number of endowment shock
    ν_s::Real = 0.95,               # scale of patience
    ν_p::Real = 0.10,               # probability of patience
    ν_size::Integer = 2,            # number of preference shock
    a_min::Real = -7.0,             # min of asset holding
    a_max::Real = 350.0,            # max of asset holding
    a_size_neg::Integer = 701,      # number of grid of negative asset holding for VFI
    a_size_pos::Integer = 51,       # number of grid of positive asset holding for VFI
    a_degree::Integer = 3,          # curvature of the positive asset gridpoints
    μ_scale::Integer = 7,           # scale governing the number of grids in computing density
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
    ν_Γ = [ν_p, 1.0 - ν_p]

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length = a_size_neg))
    a_grid_pos = ((range(0.0, stop = a_size_pos - 1, length = a_size_pos) / (a_size_pos - 1)) .^ a_degree) * a_max
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero, a_grid)[]

    # asset holding grid for μ
    # a_size_neg_μ = convert(Int, (a_size_neg-1)*μ_scale+1)
    a_size_neg_μ = convert(Int, a_size_neg)
    a_grid_neg_μ = collect(range(a_min, 0.0, length = a_size_neg_μ))
    a_size_pos_μ = convert(Int, (a_size_pos - 1) * μ_scale + 1)
    a_grid_pos_μ = collect(range(0.0, a_max, length = a_size_pos_μ))
    a_grid_μ = cat(a_grid_neg_μ, a_grid_pos_μ[2:end], dims = 1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(iszero, a_grid_μ)[]

    # compute equilibrium prices and quantities
    ξ = (1.0 - ψ) / (1 - λ - ψ)
    Λ = β_f * (1.0 - ψ + ψ * ξ)
    LR = ξ / θ
    AD = LR / (LR - 1.0)
    ι = λ * θ / Λ
    r_k = r_f + ι
    E = exp(e_SS)
    K = E * ((r_k + δ) / α)^(1.0 / (α - 1.0))
    w = (1.0 - α) * (K / E)^α

    # return values
    return (
        β = β,
        β_f = β_f,
        r_f = r_f,
        σ = σ,
        η = η,
        δ = δ,
        α = α,
        ψ = ψ,
        λ = λ,
        θ = θ,
        a_degree = a_degree,
        μ_scale = μ_scale,
        e_ρ = e_ρ,
        e_σ = e_σ,
        e_size = e_size,
        e_Γ = e_Γ,
        e_grid = e_grid,
        ν_s = ν_s,
        ν_p = ν_p,
        ν_size = ν_size,
        ν_Γ = ν_Γ,
        ν_grid = ν_grid,
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
        AD = AD,
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
    K::Real
    L::Real
    N::Real
end

mutable struct MutableVariables
    """
    construct a type for mutable variables
    """
    p::Array{Float64,2}
    q::Array{Float64,2}
    rbl::Array{Float64,2}
    V::Array{Float64,3}
    V_d::Array{Float64,2}
    V_nd::Array{Float64,3}
    policy_a::Array{Float64,3}
    threshold_a::Array{Float64,2}
    threshold_e::Array{Float64,2}
    μ::Array{Float64,3}
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

function variables_function(parameters::NamedTuple)
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack a_size, a_grid, a_size_μ, e_size, e_grid, e_ρ, e_σ, ν_size, w, r_f, ι, η, σ = parameters

    # define repayment probability, pricing function, and risky borrowing limit
    #=
    p = ones(a_size, e_size)
    q = p ./ (1.0 + r_f + ι)
    rbl = zeros(e_size, 2)
    rbl[:, 1] .= a_grid[1]
    rbl[:, 2] .= a_grid[1] / (1.0 + r_f + ι)
    =#

    p = zeros(a_size, e_size)
    q = zeros(a_size, e_size)
    rbl = zeros(e_size, 2)

    for e_i = 1:e_size
        @inbounds e_μ = e_ρ * e_grid[e_i]

        p_function(a_p) = 1.0 - cdf(LogNormal(e_μ, e_σ), -a_p / w)
        @inbounds @views p[:, e_i] = p_function.(a_grid)
        @inbounds @views q[:, e_i] = p[:, e_i] ./ (1.0 + r_f + ι)

        qa_funcion(a_p) = (p_function(a_p) / (1.0 + r_f + ι)) * a_p
        @inbounds rbl_lb, rbl_ub = min_bounds_function(qa_funcion, a_grid[1], 0.0)
        res_rbl = optimize(qa_funcion, rbl_lb, rbl_ub)
        @inbounds rbl[e_i, 1] = Optim.minimizer(res_rbl)
        @inbounds rbl[e_i, 2] = Optim.minimum(res_rbl)
    end

    # define value and policy functions
    V = zeros(a_size, e_size, ν_size)
    V_d = zeros(e_size, ν_size)
    V_nd = zeros(a_size, e_size, ν_size)
    policy_a = zeros(a_size, e_size, ν_size)

    #=
    for e_i = 1:e_size
        @inbounds y = w * exp(e_grid[e_i])
        @inbounds @views V_d[e_i, :] .= utility_function((1 - η) * y, σ)

        Threads.@threads for a_i = 1:a_size
            @inbounds a = a_grid[a_i]
            @inbounds @views V_nd[a_i, e_i, :] .= utility_function(y + a - q[a_i, e_i] * a, σ)

            @inbounds if V_d[e_i, 1] > V_nd[a_i, e_i, 1]
                @inbounds @views V[a_i, e_i, :] = V_d[e_i, :]
            else
                @inbounds @views V[a_i, e_i, :] = V_nd[a_i, e_i, :]
            end
        end
    end
    =#

    # define thresholds conditional on endowment or asset
    threshold_a = zeros(e_size, ν_size)
    threshold_e = zeros(a_size, ν_size)

    #=
    for e_i = 1:e_size
        @inbounds @views V_nd_Non_Inf = findall(V_nd[:, e_i, 1] .!= -Inf)
        @inbounds V_nd_itp = Akima(a_grid[V_nd_Non_Inf], V_nd[V_nd_Non_Inf, e_i, 1])
        @inbounds V_diff_itp(a) = V_nd_itp(a) - V_d[e_i, 1]
        @inbounds V_diff_lb, V_diff_ub = zero_bounds_function(V_d[e_i, 1], V_nd[:, e_i, 1], a_grid)
        @inbounds @views threshold_a[e_i, :] .= find_zero(a -> V_diff_itp(a), (V_diff_lb, V_diff_ub), Bisection())
    end

    @inbounds @views threshold_earning_itp = Akima(-threshold_a[:, 1], w * exp.(e_grid))
    for a_i = 1:a_size
        @inbounds earning_thres = threshold_earning_itp(-a_grid[a_i])
        e_thres = earning_thres > 0.0 ? log(earning_thres / w) : -Inf
        @inbounds @views threshold_e[a_i, :] .= e_thres
    end
    =#

    # define cross-sectional distribution
    μ_size = a_size_μ * e_size * ν_size
    μ = ones(a_size_μ, e_size, ν_size) ./ μ_size

    # define aggregate variables
    K = 0.0
    L = 0.0
    N = 0.0
    aggregate_var = MutableAggregateVariables(K, L, N)

    # return outputs
    variables = MutableVariables(p, q, rbl, V, V_d, V_nd, policy_a, threshold_a, threshold_e, μ, aggregate_var)
    return variables
end

function EV_itp_function(a_p::Real, e_i::Integer, V_d_p::Array{Float64,2}, V_nd_p::Array{Float64,3}, threshold_a::Real, parameters::NamedTuple)
    """
    construct interpolated expected value function
    """

    # unpack parameters
    @unpack a_grid, e_size, e_Γ, ν_size, ν_Γ = parameters

    # construct container
    EV = 0.0

    # loop nested functions
    for e_p_i = 1:e_size, ν_p_i = 1:ν_size

        # interpolated non-defaulting value function
        @inbounds @views V_nd_p_Non_Inf = findall(V_nd_p[:, e_p_i, ν_p_i] .!= -Inf)
        @inbounds @views a_grid_itp = a_grid[V_nd_p_Non_Inf]
        @inbounds @views V_nd_p_grid_itp = V_nd_p[V_nd_p_Non_Inf, e_p_i, ν_p_i]
        V_nd_p_itp = Akima(a_grid_itp, V_nd_p_grid_itp)

        # interpolated value function based on defaulting threshold
        @inbounds V_p_itp(a_p) = a_p >= threshold_a ? V_nd_p_itp(a_p) : V_d_p[e_p_i, ν_p_i]

        # update expected value
        @inbounds EV += ν_Γ[ν_p_i] * e_Γ[e_i, e_p_i] * V_p_itp(a_p)
    end

    # return value
    return EV
end

function value_and_policy_function!(V_d_p::Array{Float64,2}, V_nd_p::Array{Float64,3}, variables::MutableVariables, parameters::NamedTuple)
    """
    update value and policy functions
    """

    # unpack parameters
    @unpack a_size, a_grid = parameters
    @unpack ν_size, ν_grid = parameters
    @unpack e_size, e_grid = parameters
    @unpack β, σ, r_f, ι, η, w = parameters

    # loop over all states
    for e_i = 1:e_size

        # construct earning
        @inbounds y = w * exp(e_grid[e_i])

        # extract risky borrowing limit and maximum discounted borrowing amount
        @inbounds @views rbl_a, rbl_qa = variables.rbl[e_i, :]

        # construct interpolated discounted borrowing amount functions
        @inbounds @views q_e = variables.q[:, e_i]
        qa_function_itp = Akima(a_grid, q_e .* a_grid)

        for ν_i = 1:ν_size
            println("e_i = $e_i and ν_i = $ν_i")

            # extract preference
            @inbounds ν = ν_grid[ν_i]

            # compute the next-period discounted expected value funtions and interpolated functions
            EV_itp(a_p) = EV_itp_function(a_p, e_i, V_d_p, V_nd_p, variables.threshold_a[e_i, ν_i], parameters)
            V_hat_itp(a_p) = ν * β * EV_itp(a_p)

            # compute defaulting value
            @inbounds variables.V_d[e_i, ν_i] = utility_function((1 - η) * y, σ) + V_hat_itp(0.0)

            # compute non-defaulting value
            Threads.@threads for a_i = 1:a_size

                # cash on hand
                @inbounds CoH = y + a_grid[a_i]

                if (CoH - rbl_qa) >= 0.0
                    object_nd(a_p) = -(utility_function(CoH - qa_function_itp(a_p), σ) + V_hat_itp(a_p))
                    lb, ub = min_bounds_function(object_nd, rbl_a - eps(), CoH * (1 + r_f + ι))
                    res_nd = optimize(object_nd, lb, ub)
                    @inbounds variables.V_nd[a_i, e_i, ν_i] = -Optim.minimum(res_nd)
                    @inbounds variables.policy_a[a_i, e_i, ν_i] = Optim.minimizer(res_nd)
                    if variables.V_nd[a_i, e_i, ν_i] > variables.V_d[e_i, ν_i]
                        # repayment
                        @inbounds variables.V[a_i, e_i, ν_i] = variables.V_nd[a_i, e_i, ν_i]
                    else
                        # voluntary default
                        @inbounds variables.V[a_i, e_i, ν_i] = variables.V_d[e_i, ν_i]
                    end
                else
                    # involuntary default
                    @inbounds variables.V_nd[a_i, e_i, ν_i] = utility_function(0.0, σ)
                    @inbounds variables.policy_a[a_i, e_i, ν_i] = -Inf
                    @inbounds variables.V[a_i, e_i, ν_i] = variables.V_d[e_i, ν_i]
                end
            end
        end
    end
end

function threshold_function!(variables::MutableVariables, parameters::NamedTuple)
    """
    update thresholds
    """

    # unpack parameters
    @unpack ν_size, e_size, e_grid, a_size, a_grid, w = parameters

    for ν_i = 1:ν_size

        # defaulting thresholds in wealth (a)
        for e_i = 1:e_size
            @inbounds @views V_nd_Non_Inf = findall(variables.V_nd[:, e_i, ν_i] .!= -Inf)
            @inbounds @views a_grid_itp = a_grid[V_nd_Non_Inf]
            @inbounds @views V_nd_grid_itp = variables.V_nd[V_nd_Non_Inf, e_i, ν_i]
            V_nd_itp = Akima(a_grid_itp, V_nd_grid_itp)
            @inbounds V_diff_itp(a) = V_nd_itp(a) - variables.V_d[e_i, ν_i]
            @inbounds V_diff_lb, V_diff_ub = zero_bounds_function(variables.V_d[e_i, ν_i], variables.V_nd[:, e_i, ν_i], a_grid)
            @inbounds variables.threshold_a[e_i, ν_i] = find_zero(a -> V_diff_itp(a), (V_diff_lb, V_diff_ub), Bisection())
        end

        # defaulting thresholds in endowment (e)
        @inbounds @views thres_a_grid_itp = -variables.threshold_a[:, ν_i]
        earning_grid_itp = w * exp.(e_grid)
        # threshold_earning_itp = Akima(thres_a_grid_itp, earning_grid_itp)
        threshold_earning_itp = Spline1D(thres_a_grid_itp, earning_grid_itp; k = 1, bc = "extrapolate")
        for a_i = 1:a_size
            @inbounds earning_thres = threshold_earning_itp(-a_grid[a_i])
            e_thres = earning_thres > 0.0 ? log(earning_thres / w) : -Inf
            @inbounds variables.threshold_e[a_i, ν_i] = e_thres
        end
    end
end

function pricing_and_rbl_function!(variables::MutableVariables, parameters::NamedTuple)
    """
    update pricing function and borrowing risky limit
    """

    # unpack parameters
    @unpack r_f, ι, a_size, a_grid, e_size, e_grid, e_ρ, e_σ, ν_size, ν_Γ = parameters

    # initialization
    variables.p .= 0.0
    variables.q .= 0.0

    # loop over states
    for e_i = 1:e_size

        # repayment probability and pricing funciton
        Threads.@threads for a_p_i = 1:a_size
            @inbounds e_μ = e_ρ * e_grid[e_i]
            for ν_p_i = 1:ν_size
                @inbounds variables.p[a_p_i, e_i] += ν_Γ[ν_p_i] * (1.0 - cdf(Normal(e_μ, e_σ), variables.threshold_e[a_p_i, ν_p_i]))
                @inbounds variables.q[a_p_i, e_i] += ν_Γ[ν_p_i] * variables.p[a_p_i, e_i] / (1.0 + r_f + ι)
            end
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

function solve_function!(variables::MutableVariables, parameters::NamedTuple; tol::Real = tol, iter_max::Integer = iter_max)
    """
    solve stationary equlibrium using one-loop algorithm
    """

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving stationary equlibrium (one-loop): ")

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
        value_and_policy_function!(V_d_p, V_nd_p, variables, parameters)

        # thresholds
        threshold_function!(variables, parameters)

        # pricing function and borrowing risky limit
        pricing_and_rbl_function!(variables, parameters)

        # check convergence
        V_crit = norm(variables.V .- V_p, Inf)
        q_crit = norm(variables.q .- q_p, Inf)
        crit = max(V_crit, q_crit)

        # report progress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end

#=================#
# Solve the model #
#=================#
parameters = parameters_function()
variables = variables_function(parameters)
solve_function!(variables, parameters; tol = 1E-8, iter_max = 10)

e_label = round.(exp.(parameters.e_grid), digits = 2)'
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :], legend = :topleft, label = e_label)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :] .* parameters.a_grid_neg, legend = :topleft, label = e_label)
plot!(variables.rbl[:, 1], variables.rbl[:, 2], label = "rbl", seriestype = :scatter)
plot!(parameters.a_grid_neg, parameters.a_grid_neg, lc = :black, label = "")

plot(parameters.a_grid, variables.q[1:parameters.a_size, :] .* parameters.a_grid, legend = :bottomright, label = e_label)
plot!(variables.rbl[:, 1], variables.rbl[:, 2], label = "rbl", seriestype = :scatter)
plot!(parameters.a_grid, parameters.a_grid, lc = :black, label = "")

plot(parameters.a_grid_neg, variables.V[1:parameters.a_ind_zero, :, 2], legend = :bottomleft, label = e_label)
plot!(variables.threshold_a[:, 2], variables.V_d[:, 2], label = "defaulting debt level", seriestype = :scatter)

plot(parameters.a_grid, variables.V[:, :, 2], legend = :bottomleft, label = e_label)
plot!(variables.threshold_a[:, 2], variables.V_d[:, 2], label = "defaulting debt level", seriestype = :scatter)
hline!([0.0], lc = :black)

plot(-variables.threshold_a[:, 1], parameters.w * exp.(parameters.e_grid), legend = :none, markershape = :circle, xlabel = "defaulting debt level", ylabel = "w*exp(e)")

plot(parameters.a_grid_neg, variables.threshold_e[1:parameters.a_ind_zero, 1], legend = :none, xlabel = "debt level", ylabel = "defaulting e level")
plot!(variables.threshold_a[:, 1], parameters.e_grid, seriestype = :scatter)

plot(parameters.a_grid_neg, parameters.w .* exp.(variables.threshold_e[1:parameters.a_ind_zero, 1]), legend = :none, xlabel = "debt level", ylabel = "defaulting w*exp(e) level")
plot!(variables.threshold_a[:, 1], parameters.w * exp.(parameters.e_grid), seriestype = :scatter)
