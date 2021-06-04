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
    e_size::Integer = 9,            # number of endowment shock
    ν_s::Real = 0.95,               # scale of patience
    ν_p::Real = 0.10,               # probability of patience
    ν_size::Integer = 2,            # number of preference shock
    a_min::Real = -5.0,             # min of asset holding
    a_max::Real = 350.0,            # max of asset holding
    a_size_neg::Integer = 501,      # number of grid of negative asset holding for VFI
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

function optim_bounds_function(obj::Function, grid_min::Real, grid_max::Real; grid_length::Integer = 50, obj_range::Integer = 1)
    """
    compute bounds for optimization
    """

    grid = range(grid_min, grid_max, length = grid_length)
    grid_size = length(grid)
    obj_grid = obj.(grid)
    obj_index = argmin(obj_grid)
    # obj_index = findfirst(obj_grid .== minimum(obj_grid))
    if obj_index < (1 + obj_range)
        lb = grid_min
        ub = grid[obj_index+obj_range]
    elseif obj_index > (grid_size - obj_range)
        lb = grid[obj_index-obj_range]
        ub = grid_max
    else
        lb = grid[obj_index-obj_range]
        ub = grid[obj_index+obj_range]
    end
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
    p = zeros(a_size, e_size)
    q = zeros(a_size, e_size)
    rbl = zeros(e_size, 2)

    for e_i = 1:e_size
        @inbounds e_μ = e_ρ * e_grid[e_i]

        p_function(a_p) = 1.0 - cdf(LogNormal(e_μ, e_σ), -a_p / w)
        @inbounds @views p[:, e_i] = p_function.(a_grid)
        @inbounds @views q[:, e_i] = p[:, e_i] ./ (1.0 + r_f + ι)

        object_rbl(a_p) = (p_function(a_p) / (1.0 + r_f + ι)) * a_p
        @inbounds rbl_lb, rbl_ub = optim_bounds_function(object_rbl, a_grid[1], 0.0)
        res_rbl = optimize(object_rbl, rbl_lb, rbl_ub)
        @inbounds rbl[e_i, 1] = Optim.minimizer(res_rbl)
        @inbounds rbl[e_i, 2] = Optim.minimum(res_rbl)
    end

    # define value and policy functions
    V = zeros(a_size, e_size, ν_size)
    V_d = zeros(e_size, ν_size)
    V_nd = zeros(a_size, e_size, ν_size)
    policy_a = zeros(a_size, e_size, ν_size)

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

    # define thresholds conditional on endowment or asset
    threshold_a = zeros(e_size, ν_size)
    threshold_e = zeros(a_size, ν_size)

    for e_i = 1:e_size
        @inbounds @views V_nd_Non_Inf = findall(V_nd[:, e_i, 1] .!= -Inf)
        @inbounds V_nd_itp = Akima(a_grid[V_nd_Non_Inf], V_nd[V_nd_Non_Inf, e_i, 1])
        @inbounds V_diff_itp(a) = V_nd_itp(a) - V_d[e_i, 1]
        @inbounds @views V_diff_lb = a_grid[minimum(findall(V_nd[:, e_i, 1] .> V_d[e_i, 1]))]
        @inbounds @views V_diff_ub = a_grid[maximum(findall(V_nd[:, e_i, 1] .< V_d[e_i, 1]))]
        @inbounds @views threshold_a[e_i, :] .= find_zero(a -> V_diff_itp(a), (V_diff_lb, V_diff_ub), Bisection())
    end

    @inbounds @views threshold_earning_itp = Akima(-threshold_a[:, 1], w * exp.(e_grid))
    for a_i = 1:a_size
        @inbounds earning_thres = threshold_earning_itp(-a_grid[a_i])
        e_thres = earning_thres > 0.0 ? log(earning_thres / w) : -Inf
        @inbounds @views threshold_e[a_i, :] .= e_thres
    end

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



function EV_itp_function(a_p::Real, e_i::Integer, V_d_p::Array{Float64,2}, V_nd_p::Array{Float64,3}, threshold_e::Array{Float64,1}, parameters::NamedTuple)
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
        V_nd_p_Non_Inf = findall(V_nd_p[:, e_p_i, ν_p_i] .!= -Inf)
        V_nd_p_itp = Akima(a_grid[V_nd_p_Non_Inf], V_nd_p[V_nd_p_Non_Inf, e_p_i, ν_p_i])

        # interpolated value function based on defaulting threshold
        V_p_itp(a_p) = a_p >= threshold_e[e_i] ? V_nd_p_itp(a_p) : V_d_p[e_p_i, ν_p_i]

        # update expected value
        EV += ν_Γ[ν_p_i] * e_Γ[e_i, e_p_i] * V_p_itp(a_p)
    end
    return EV
end



function solve_ED_function(parameters::NamedTuple; tol::Real = tol, iter_max::Integer = iter_max)
    """
    solve the economy where enforced defualt (ED) is imposed based on income
    """

    # unpack parameters
    @unpack a_size, a_size_neg, a_grid, a_grid_neg, a_ind_zero = parameters
    @unpack ν_size, ν_grid, ν_Γ = parameters
    @unpack e_size, e_grid, e_Γ, e_ρ, e_σ = parameters
    @unpack β, σ, r_f, ι, η, w = parameters

    # update household's problem
    for e_i = 1:e_size

        # extract endowment
        @inbounds e = e_grid[e_i]
        @inbounds @views rbl_a, rbl_qa = variables.rbl[e_i, :]

        # compute the next-period discounted expected value funtions and interpolated functions
        EV_itp(a_p) = EV_itp_function(a_p, e_i, V_d_p, V_nd_p, γ, parameters)
        V_hat_impatient_itp(a_p) = ν_grid[1] * β * EV_itp(a_p)
        V_hat_patient_itp(a_p) = ν_grid[2] * β * EV_itp(a_p)

        # compute defaulting value
        @inbounds variables.V_d[e_i, 1] = utility_function((1 - η) * w * exp(e), σ) + V_hat_impatient_itp(0.0)
        @inbounds variables.V_d[e_i, 2] = utility_function((1 - η) * w * exp(e), σ) + V_hat_patient_itp(0.0)

        #
        @inbounds @views q_function = Akima(a_grid, variables.q[:, e_i])
        qa_function(a_p) = q_function(a_p) * a_p

        # compute non-defaulting value
        Threads.@threads for a_i = 1:a_size

            # cash on hand
            @inbounds CoH = w * exp(e) + a_grid[a_i]

            if (CoH - rbl_qa) >= 0.0

                # impatient household
                object_nd_impatient(a_p) = -(utility_function(CoH - qa_function(a_p), σ) + V_hat_impatient_itp(a_p))
                impatient_lb, impatient_ub = optim_bounds_function(object_nd_impatient, rbl_a, CoH * (1 + r_f + ι))
                res_nd_impatient = optimize(object_nd_impatient, impatient_lb, impatient_ub)
                @inbounds variables.V_nd[a_i, e_i, 1] = -Optim.minimum(res_nd_impatient)
                @inbounds variables.policy_a[a_i, e_i, 1] = Optim.minimizer(res_nd_impatient)
                if variables.V_nd[a_i, e_i, 1] > variables.V_d[e_i, 1]
                    variables.V[a_i, e_i, 1] = variables.V_nd[a_i, e_i, 1]
                else
                    variables.V[a_i, e_i, 1] = variables.V_d[e_i, 1]
                end

                # patient household
                object_nd_patient(a_p) = -(utility_function(CoH - qa_function(a_p), σ) + V_hat_patient_itp(a_p))
                patient_lb, patient_ub = optim_bounds_function(object_nd_patient, rbl_a, CoH * (1 + r_f + ι))
                res_nd_patient = optimize(object_nd_patient, patient_lb, patient_ub)
                @inbounds variables.V_nd[a_i, e_i, 2] = -Optim.minimum(res_nd_patient)
                @inbounds variables.policy_a[a_i, e_i, 2] = Optim.minimizer(res_nd_patient)
                if variables.V_nd[a_i, e_i, 2] > variables.V_d[e_i, 2]
                    variables.V[a_i, e_i, 2] = variables.V_nd[a_i, e_i, 2]
                else
                    variables.V[a_i, e_i, 2] = variables.V_d[e_i, 2]
                end
            else
                @inbounds @views variables.V_nd[a_i, e_i, :] .= utility_function(0.0, σ)
                @inbounds @views variables.policy_a[a_i, e_i, :] .= -Inf
                @inbounds @views variables.V[a_i, e_i, :] = variables.V_d[e_i, :]
            end
        end
    end
end

function value_function!(V_p::Array{Float64,3}, q_p::Array{Float64,2}, variables::MutableVariables, parameters::NamedTuple; slow_updating::Real = 1.0)
    """
    update value and policy functions
    """

    # unpack parameters
    @unpack a_grid, a_size, e_grid, e_size, e_Γ, ν_grid, ν_size, ν_Γ = parameters

    # feasible set and conditional value function
    Threads.@threads for e_i = 1:e_size
        @inbounds e = e_grid[e_i]
        @inbounds ν = ν_grid[β_i]

        # compute the next-period discounted expected value funtions and interpolated functions
        @inbounds @views V_expt_p = (ν_Γ[1] * V_p[:, :, 1] + ν_Γ[2] * V_p[:, :, 2]) * e_Γ[e_i, :]
        @inbounds @views V_hat_impatient = ν_grid[1] * β * V_expt_p
        @inbounds @views V_hat_patient = ν_grid[2] * β * V_expt_p
        V_hat_impatient_itp = Akima(a_grid, V_hat_impatient)
        V_hat_patient_itp = Akima(a_grid, V_hat_patient)

        # compute defaulting value
        variables.V_d[e_i, 1] = utility_function((1 - η) * w * exp(e), σ) + V_hat_impatient[a_ind_zero]
        variables.V_d[e_i, 2] = utility_function((1 - η) * w * exp(e), σ) + V_hat_patient[a_ind_zero]

        q = q_p[:, e_i]
        qa = [q .* a_grid_neg; a_grid_pos[2:end]]
        qa_itp = Akima(a_grid, qa)



        if c <= 0.0 || (action_i == 1 && a_i >= a_ind_zero)
            @inbounds variables.v[action_i, β_i, e_i, a_i, s_i] = -Inf
        else
            W_expect = 0.0
            for e_p_i = 1:e_size, β_p_i = 1:β_size
                if action_i == 1
                    @inbounds W_expect += β_Γ[β_i, β_p_i] * e_Γ[e_i, e_p_i] * W_p[β_p_i, e_p_i, a_p_i, 1]
                else
                    for s_p_i = 1:s_size
                        @inbounds W_expect += β_Γ[β_i, β_p_i] * e_Γ[e_i, e_p_i] * variables.Q_s[s_p_i, action_i, e_i, a_i, s_i] * W_p[β_p_i, e_p_i, a_p_i, s_p_i]
                    end
                end
            end
            @inbounds variables.v[action_i, β_i, e_i, a_i, s_i] = utility_function(c, γ) + β * ρ * W_expect
        end
    end

    # unconditional value function and choice probability
    Threads.@threads for s_i = 1:s_size
        for a_i = 1:a_size, e_i = 1:e_size, β_i = 1:β_size
            @inbounds @views V = sum(exp.(variables.v[:, β_i, e_i, a_i, s_i] ./ α))
            @inbounds variables.W[β_i, e_i, a_i, s_i] = α * log(V)
            @inbounds @views variables.σ[:, β_i, e_i, a_i, s_i] = exp.(variables.v[:, β_i, e_i, a_i, s_i] ./ α) ./ V
            @inbounds @views variables.F[:, β_i, e_i, a_i, s_i] .= 1.0
            @inbounds @views variables.F[variables.σ[:, β_i, e_i, a_i, s_i].≈0.0, β_i, e_i, a_i, s_i] .= 0.0

            @inbounds @views V_ND = sum(exp.(variables.v[2:end, β_i, e_i, a_i, s_i] ./ α))
            if V_ND ≈ 0.0
                @inbounds @views variables.σ_ND[:, β_i, e_i, a_i, s_i] .= 0.0
            else
                @inbounds @views variables.σ_ND[:, β_i, e_i, a_i, s_i] = exp.(variables.v[2:end, β_i, e_i, a_i, s_i] ./ α) ./ V_ND
            end
            @inbounds @views variables.F_ND[:, β_i, e_i, a_i, s_i] .= 1.0
            @inbounds @views variables.F_ND[variables.σ_ND[:, β_i, e_i, a_i, s_i].≈0.0, β_i, e_i, a_i, s_i] .= 0.0
        end
    end

    if slow_updating != 1.0
        variables.W = slow_updating * variables.W + (1.0 - slow_updating) * W_p
    end
end

#=================#
# Solve the model #
#=================#
parameters = parameters_function()
variables = variables_function(parameters)

e_label = round.(exp.(parameters.e_grid), digits = 2)'
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :], legend = :topleft, label = e_label)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :] .* parameters.a_grid_neg, legend = :topleft, label = e_label)
plot!(variables.rbl[:, 1], variables.rbl[:, 2], label = "rbl", seriestype = :scatter)

plot(parameters.a_grid_neg, variables.V[1:parameters.a_ind_zero, :, 1], legend = :bottomleft, label = e_label)
plot!(variables.threshold_a[:, 1], variables.V_d[:, 1], label = "defaulting debt level", seriestype = :scatter)

plot(-variables.threshold_a[:, 1], parameters.w * exp.(parameters.e_grid), legend = :none, markershape = :circle, xlabel = "defaulting debt level", ylabel = "w*exp(e)")

plot(parameters.a_grid_neg, variables.threshold_e[1:parameters.a_ind_zero, 1], legend = :none, xlabel = "debt level", ylabel = "defaulting e level")
plot!(variables.threshold_a[:, 1], parameters.e_grid, seriestype = :scatter)

plot(parameters.a_grid_neg, parameters.w .* exp.(variables.threshold_e[1:parameters.a_ind_zero, 1]), legend = :none, xlabel = "debt level", ylabel = "defaulting w*exp(e) level")
plot!(variables.threshold_a[:, 1], parameters.w * exp.(parameters.e_grid), seriestype = :scatter)
