#===========================#
# Import packages and files #
#===========================#
using Distributions
using QuadGK
using JLD2: @save, @load
using LinearAlgebra
using Parameters: @unpack
using PrettyTables
using ProgressMeter
using QuantEcon: stationary_distributions, MarkovChain # gridmake, rouwenhorst, tauchen
using CSV
using Tables

# print out the number of threads
println("Julia is running with $(Threads.nthreads()) threads...")

#==================#
# Define functions #
#==================#
function adda_cooper(N::Integer, ρ::Real, σ::Real; μ::Real = 0.0)
    """
    Approximation of an autoregression process with a Markov chain proposed by Adda and Cooper (2003)
    """

    σ_ϵ = σ / sqrt(1.0 - ρ^2.0)
    ϵ = σ_ϵ .* quantile.(Normal(), [i / N for i = 0:N]) .+ μ
    z = zeros(N)
    for i = 1:N
        if i != (N + 1) / 2
            z[i] = N * σ_ϵ * (pdf(Normal(), (ϵ[i] - μ) / σ_ϵ) - pdf(Normal(), (ϵ[i+1] - μ) / σ_ϵ)) + μ
        end
    end
    Π = zeros(N, N)
    if ρ == 0.0
        Π .= 1.0 / N
    else
        for i = 1:N, j = 1:N
            f(u) = exp(-(u - μ)^2.0 / (2.0 * σ_ϵ^2.0)) * (cdf(Normal(), (ϵ[j+1] - μ * (1.0 - ρ) - ρ * u) / σ) - cdf(Normal(), (ϵ[j] - μ * (1.0 - ρ) - ρ * u) / σ))
            integral, err = quadgk(u -> f(u), ϵ[i], ϵ[i+1])
            Π[i, j] = (N / sqrt(2.0 * π * σ_ϵ^2.0)) * integral
        end
    end
    return z, Π
end

function parameters_function(;
    β::Real = 0.93 / 0.98,          # discount factor (households)
    ρ::Real = 0.98,                 # survival rate
    r_f::Real = 0.04,               # risk-free rate # 1.04*ρ-1.0
    β_f::Real = 1.0 / (1.0 + r_f),  # discount factor (bank)
    τ::Real = 0.04,                 # transaction cost
    σ::Real = 2.00,                 # CRRA coefficient
    η::Real = 0.25,                 # garnishment rate
    δ::Real = 0.08,                 # depreciation rate
    α::Real = 0.36,                 # capital share
    ψ::Real = 1.0 - 1.0 / 10.0,     # exogenous retention ratio
    θ::Real = 1.0 / 3.0,            # diverting fraction
    p_h::Real = 1.0 / 7.0,          # prob. of history erased
    ζ_a::Real = 0.0055,             # EV scale parameter (asset choice)
    ζ_d::Real = 0.18,               # EV scale parameter (default)
    e_1_σ::Real = 0.448,            # s.d. of permanent endowment shock
    e_1_size::Integer = 2,          # number of permanent endowment shock
    e_2_ρ::Real = 0.957,            # AR(1) of persistent endowment shock
    e_2_σ::Real = 0.129,            # s.d. of persistent endowment shock
    e_2_size::Integer = 3,          # number of persistent endowment shock
    e_3_σ::Real = 0.351,            # s.d. of transitory endowment shock
    e_3_size::Integer = 3,          # number oftransitory endowment shock
    a_min::Real = -10.0,            # min of asset holding
    a_max::Real = 80.0,             # max of asset holding
    a_size_neg::Integer = 101,      # number of grid of negative asset holding for VFI
    a_size_pos::Integer = 51,       # number of grid of positive asset holding for VFI
    a_degree::Integer = 2,          # curvature of the positive asset gridpoints
)
    """
    contruct an immutable object containg all paramters
    """

    # permanent endowment shock
    # e_1_grid, e_1_Γ = adda_cooper(e_1_size, 0.0, e_1_σ)
    e_1_grid = [-e_1_σ, e_1_σ]
    e_1_Γ = Matrix(1.0I, e_1_size, e_1_size)
    G_e_1 = [1.0 / e_1_size for i = 1:e_1_size]

    # persistent endowment shock
    # e_2_MC = tauchen(e_2_size, e_2_ρ, e_2_σ, 0.0, 3)
    # e_2_MC = rouwenhorst(e_2_size, e_2_ρ, e_2_σ, 0.0)
    # e_2_Γ = e_2_MC.p
    # e_2_grid = collect(e_2_MC.state_values)
    e_2_grid, e_2_Γ = adda_cooper(e_2_size, e_2_ρ, e_2_σ)
    G_e_2 = stationary_distributions(MarkovChain(e_2_Γ, e_2_grid))[1]

    # transitory endowment shock
    # e_3_grid, e_3_Γ = adda_cooper(e_3_size, 0.0, e_3_σ)
    e_3_bar = sqrt((3 / 2) * e_3_σ^2)
    e_3_grid = [-e_3_bar, 0.0, e_3_bar]
    e_3_Γ = [1.0 / e_3_size for i = 1:e_3_size]
    G_e_3 = e_3_Γ

    # aggregate labor endowment
    E = 1.0

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length = a_size_neg))
    a_grid_pos_1 = collect(range(0.0, -a_min, length = a_size_neg))
    # a_grid_pos_2 = collect(range(-a_min, a_max, length = a_size_pos))
    a_grid_pos_2 = ((range(0.0, stop = a_size_pos - 1, length = a_size_pos) / (a_size_pos - 1)) .^ a_degree) * (a_max + a_min) .- a_min
    a_grid_pos = cat(a_grid_pos_1[1:(end-1)], a_grid_pos_2, dims = 1)
    a_size_pos = length(a_grid_pos)
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero, a_grid)[]

    # return values
    return (
        β = β,
        ρ = ρ,
        r_f = r_f,
        β_f = β_f,
        τ = τ,
        σ = σ,
        η = η,
        δ = δ,
        α = α,
        ψ = ψ,
        θ = θ,
        p_h = p_h,
        ζ_a = ζ_a,
        ζ_d = ζ_d,
        e_1_σ = e_1_σ,
        e_1_size = e_1_size,
        e_1_Γ = e_1_Γ,
        e_1_grid = e_1_grid,
        G_e_1 = G_e_1,
        e_2_ρ = e_2_ρ,
        e_2_σ = e_2_σ,
        e_2_size = e_2_size,
        e_2_Γ = e_2_Γ,
        e_2_grid = e_2_grid,
        G_e_2 = G_e_2,
        e_3_σ = e_3_σ,
        e_3_size = e_3_size,
        e_3_Γ = e_3_Γ,
        e_3_grid = e_3_grid,
        G_e_3 = G_e_3,
        E = E,
        a_grid = a_grid,
        a_grid_neg = a_grid_neg,
        a_grid_pos = a_grid_pos,
        a_size = a_size,
        a_size_neg = a_size_neg,
        a_size_pos = a_size_pos,
        a_ind_zero = a_ind_zero,
        a_degree = a_degree,
    )
end

mutable struct Mutable_Aggregate_Prices
    """
    construct a type for mutable aggregate prices
    """
    λ::Real
    ξ_λ::Real
    Λ_λ::Real
    leverage_ratio_λ::Real
    KL_to_D_ratio_λ::Real
    ι_λ::Real
    r_k_λ::Real
    K_λ::Real
    w_λ::Real
end

mutable struct Mutable_Aggregate_Variables
    """
    construct a type for mutable aggregate variables
    """
    K::Real
    L::Real
    D::Real
    N::Real
    leverage_ratio::Real
    KL_to_D_ratio::Real
    debt_to_earning_ratio::Real
    share_of_filers::Real
    share_in_debts::Real
    avg_loan_rate::Real
    avg_loan_rate_pw::Real
end

mutable struct Mutable_Variables
    """
    construct a type for mutable variables
    """
    aggregate_prices::Mutable_Aggregate_Prices
    aggregate_variables::Mutable_Aggregate_Variables
    R::Array{Float64,3}
    q::Array{Float64,3}
    V::Array{Float64,4}
    V_d::Array{Float64,4}
    V_nd::Array{Float64,4}
    V_nd_all::Array{Float64,5}
    V_pos::Array{Float64,4}
    V_pos_all::Array{Float64,5}
    policy_a::Array{Float64,5}
    policy_d::Array{Float64,4}
    policy_pos_a::Array{Float64,5}
    μ::Array{Float64,5}
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

function aggregate_prices_λ_funtion(parameters::NamedTuple; λ::Real)
    """
    compute aggregate prices for given incentive multiplier λ
    """
    @unpack ρ, α, ψ, β_f, θ, r_f, δ, E = parameters

    ξ_λ = (1.0 - ψ) / (1.0 - λ - ψ)
    Λ_λ = β_f * (1.0 - ψ + ψ * ξ_λ)
    leverage_ratio_λ = ξ_λ / θ
    KL_to_D_ratio_λ = leverage_ratio_λ / (leverage_ratio_λ - 1.0)
    ι_λ = λ * θ / Λ_λ
    r_k_λ = r_f + ι_λ
    K_λ = E * ((r_k_λ + δ) / α)^(1.0 / (α - 1.0))
    w_λ = (1.0 - α) * (K_λ / E)^α

    return ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ
end

function variables_function(parameters::NamedTuple; λ::Real)
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack a_ind_zero, a_size, a_grid, a_size_pos, a_size_neg, a_grid_neg, e_1_size, e_1_grid, e_1_Γ, e_2_size, e_2_grid, e_2_Γ, e_2_ρ, e_2_σ, e_3_size, e_3_grid, e_3_Γ, ρ, r_f, τ = parameters

    # define aggregate prices and variables
    ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ = aggregate_prices_λ_funtion(parameters; λ = λ)
    K = K_λ
    L = 0.0
    D = 0.0
    N = 0.0
    leverage_ratio = 0.0
    KL_to_D_ratio = 0.0
    debt_to_earning_ratio = 0.0
    share_of_filers = 0.0
    share_in_debts = 0.0
    avg_loan_rate = 0.0
    avg_loan_rate_pw = 0.0
    aggregate_prices = Mutable_Aggregate_Prices(λ, ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ)
    aggregate_variables = Mutable_Aggregate_Variables(K, L, D, N, leverage_ratio, KL_to_D_ratio, debt_to_earning_ratio, share_of_filers, share_in_debts, avg_loan_rate, avg_loan_rate_pw)

    # define repayment probability, pricing function, and risky borrowing limit
    R = zeros(a_size_neg, e_1_size, e_2_size)
    q = ones(a_size, e_1_size, e_2_size) .* ρ ./ (1.0 + r_f)

    # define value and policy functions
    V = zeros(a_size, e_1_size, e_2_size, e_3_size)
    V_d = zeros(a_size, e_1_size, e_2_size, e_3_size)
    V_nd = zeros(a_size, e_1_size, e_2_size, e_3_size)
    V_nd_all = zeros(a_size, a_size, e_1_size, e_2_size, e_3_size)
    V_pos = zeros(a_size_pos, e_1_size, e_2_size, e_3_size)
    V_pos_all = zeros(a_size_pos, a_size_pos, e_1_size, e_2_size, e_3_size)
    policy_a = zeros(a_size, a_size, e_1_size, e_2_size, e_3_size)
    policy_d = zeros(a_size, e_1_size, e_2_size, e_3_size)
    policy_pos_a = zeros(a_size_pos, a_size_pos, e_1_size, e_2_size, e_3_size)

    # define cross-sectional distribution
    μ = zeros(a_size, e_1_size, e_2_size, e_3_size, 2)
    μ_size = (a_size + a_size_pos) * e_1_size * e_2_size * e_3_size
    μ[:, :, :, :, 1] .= 1.0 / μ_size
    μ[a_ind_zero:end, :, :, :, 2] .= 1.0 / μ_size

    # return outputs
    variables = Mutable_Variables(aggregate_prices, aggregate_variables, R, q, V, V_d, V_nd, V_nd_all, V_pos, V_pos_all, policy_a, policy_d, policy_pos_a, μ)
    return variables
end

function EV_function(V_p::Array{Float64,4}, parameters::NamedTuple)
    """
    precompute the expected value function
    """

    # unpack parameters
    @unpack e_1_size, e_1_Γ, e_2_size, e_2_Γ, e_3_size, e_3_Γ = parameters

    # construct container
    a_size_ = size(V_p)[1]
    EV = zeros(a_size_, e_1_size, e_2_size)

    for e_2_i = 1:e_2_size, e_1_i = 1:e_1_size
        for e_3_p_i = 1:e_3_size, e_2_p_i = 1:e_2_size, e_1_p_i = 1:e_1_size
            @inbounds @views EV[:, e_1_i, e_2_i] += e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * V_p[:, e_1_p_i, e_2_p_i, e_3_p_i]
        end
    end

    # repalce NaN with -Inf
    replace!(EV, NaN => -Inf)

    # return value
    return EV
end

function value_and_policy_function(V_p::Array{Float64,4}, V_d_p::Array{Float64,4}, V_nd_p::Array{Float64,4}, V_pos_p::Array{Float64,4}, q::Array{Float64,3}, w::Real, parameters::NamedTuple; slow_updating::Real = 1.0)
    """
    one-step update of value and policy functions
    """

    # unpack parameters
    @unpack a_size, a_grid, a_size_pos, a_grid_pos, a_ind_zero, e_1_size, e_1_grid, e_1_Γ, e_2_size, e_2_grid, e_2_Γ, e_3_size, e_3_grid, e_3_Γ, ρ, β, σ, η, r_f, p_h, ζ_a, ζ_d = parameters

    # construct containers
    V = zeros(a_size, e_1_size, e_2_size, e_3_size)
    V_d = zeros(a_size, e_1_size, e_2_size, e_3_size)
    V_nd = zeros(a_size, e_1_size, e_2_size, e_3_size)
    V_nd_all = zeros(a_size, a_size, e_1_size, e_2_size, e_3_size)
    V_pos = zeros(a_size_pos, e_1_size, e_2_size, e_3_size)
    V_pos_all = zeros(a_size_pos, a_size_pos, e_1_size, e_2_size, e_3_size)
    policy_a = zeros(a_size, a_size, e_1_size, e_2_size, e_3_size)
    policy_d = zeros(a_size, e_1_size, e_2_size, e_3_size)
    policy_pos_a = zeros(a_size_pos, a_size_pos, e_1_size, e_2_size, e_3_size)

    # precomputation
    V_hat = ρ * β * EV_function(V_p, parameters)
    V_hat_pos = ρ * β * EV_function(V_pos_p, parameters)

    # loop over all states
    for e_3_i = 1:e_3_size, e_2_i = 1:e_2_size, e_1_i = 1:e_1_size, a_i = 1:a_size

        # construct states
        @inbounds a = a_grid[a_i]
        @inbounds y = w * exp(e_1_grid[e_1_i] + e_2_grid[e_2_i] + e_3_grid[e_3_i])

        # construct interpolated discounted borrowing amount functions
        @inbounds @views qa = q[:, e_1_i, e_2_i] .* a_grid

        # compute defaulting value
        c_d = (1.0 - η) * y
        # @inbounds V_d[a_i, e_1_i, e_2_i, e_3_i] = a < -η * y ? (1.0 - ρ * β) * utility_function(c_d, σ) + V_hat_pos[1, e_1_i, e_2_i] : -Inf
        @inbounds V_d[a_i, e_1_i, e_2_i, e_3_i] = a < -η * y ? utility_function(c_d, σ) + V_hat_pos[1, e_1_i, e_2_i] : -Inf

        # compute non-defaulting value
        c_nd = y .+ a .- qa
        # @inbounds @views V_nd_all[:, a_i, e_1_i, e_2_i, e_3_i] = (1.0 - ρ * β) * utility_function.(c_nd, σ) .+ V_hat[:, e_1_i, e_2_i]
        @inbounds @views V_nd_all[:, a_i, e_1_i, e_2_i, e_3_i] = utility_function.(c_nd, σ) .+ V_hat[:, e_1_i, e_2_i]
        @inbounds @views V_nd_all[1:(argmin(qa)-1), a_i, e_1_i, e_2_i, e_3_i] .= -Inf

        @inbounds @views V_nd_all_max = maximum(V_nd_all[:, a_i, e_1_i, e_2_i, e_3_i])
        if V_nd_all_max == -Inf
            V_nd[a_i, e_1_i, e_2_i, e_3_i] = V_nd_all_max
            policy_a[a_ind_zero, a_i, e_1_i, e_2_i, e_3_i] = 1.0
        else
            V_nd_all_exp = exp.((V_nd_all[:, a_i, e_1_i, e_2_i, e_3_i] .- V_nd_all_max) ./ ζ_a)
            V_nd_all_sum = sum(V_nd_all_exp)
            V_nd[a_i, e_1_i, e_2_i, e_3_i] = V_nd_all_max + ζ_a * log(V_nd_all_sum)
            policy_a[:, a_i, e_1_i, e_2_i, e_3_i] = V_nd_all_exp ./ V_nd_all_sum
        end

        # whether to default
        V_max = max(V_nd[a_i, e_1_i, e_2_i, e_3_i], V_d[a_i, e_1_i, e_2_i, e_3_i])
        if V_max == -Inf
            V[a_i, e_1_i, e_2_i, e_3_i] = V_max
            policy_d[a_i, e_1_i, e_2_i, e_3_i] = 1.0
        else
            V_sum = exp((V_nd[a_i, e_1_i, e_2_i, e_3_i] - V_max) / ζ_d) + exp((V_d[a_i, e_1_i, e_2_i, e_3_i] - V_max) / ζ_d)
            V[a_i, e_1_i, e_2_i, e_3_i] = V_max + ζ_d * log(V_sum)
            policy_d[a_i, e_1_i, e_2_i, e_3_i] = exp((V_d[a_i, e_1_i, e_2_i, e_3_i] - V_max) / ζ_d) / V_sum
        end

        # bad credit history
        if a_i >= a_ind_zero
            a_pos_i = a_i - a_ind_zero + 1
            @inbounds @views c_pos = y .+ a .- qa[a_ind_zero:end]
            # @inbounds @views V_pos_all[:, a_pos_i, e_1_i, e_2_i, e_3_i] = (1.0 - ρ * β) * utility_function.(c_pos, σ) .+ (p_h * V_hat[a_ind_zero:end, e_1_i, e_2_i] .+ (1.0 - p_h) * V_hat_pos[:, e_1_i, e_2_i])
            @inbounds @views V_pos_all[:, a_pos_i, e_1_i, e_2_i, e_3_i] = utility_function.(c_pos, σ) .+ (p_h * V_hat[a_ind_zero:end, e_1_i, e_2_i] .+ (1.0 - p_h) * V_hat_pos[:, e_1_i, e_2_i])
            @inbounds @views V_pos_max = maximum(V_pos_all[:, a_pos_i, e_1_i, e_2_i, e_3_i])
            if V_pos_max == -Inf
                V_pos[a_pos_i, e_1_i, e_2_i, e_3_i] = V_pos_max
                policy_pos_a[1, a_pos_i, e_1_i, e_2_i, e_3_i] = 1.0
            else
                V_pos_exp = exp.((V_pos_all[:, a_pos_i, e_1_i, e_2_i, e_3_i] .- V_pos_max) ./ ζ_a)
                V_pos_sum = sum(V_pos_exp)
                V_pos[a_pos_i, e_1_i, e_2_i, e_3_i] = V_pos_max + ζ_a * log(V_pos_sum)
                policy_pos_a[:, a_pos_i, e_1_i, e_2_i, e_3_i] = V_pos_exp ./ V_pos_sum
            end
        end
    end

    # slow updating
    if slow_updating != 1.0
        V = slow_updating * V + (1.0 - slow_updating) * V_p
        V_d = slow_updating * V_d + (1.0 - slow_updating) * V_d_p
        V_nd = slow_updating * V_nd + (1.0 - slow_updating) * V_nd_p
        V_pos = slow_updating * V_pos + (1.0 - slow_updating) * V_pos_p
    end

    # return results
    return V, V_d, V_nd, V_pos, policy_a, policy_d, policy_pos_a
end

function pricing_and_rbl_function(policy_d::Array{Float64,4}, w::Real, ι::Real, parameters::NamedTuple)
    """
    update pricing function and borrowing risky limit
    """

    # unpack parameters
    @unpack ρ, r_f, τ, η, a_ind_zero, a_size, a_grid, a_size_neg, a_grid_neg, e_1_size, e_1_grid, e_1_Γ, e_2_size, e_2_grid, e_2_Γ, e_3_size, e_3_grid, e_3_Γ = parameters

    # contruct containers
    R = zeros(a_size_neg, e_1_size, e_2_size)
    q = ones(a_size, e_1_size, e_2_size) .* ρ ./ (1.0 + r_f)

    # loop over states
    for e_2_i = 1:e_2_size, e_1_i = 1:e_1_size, a_p_i = 1:(a_size_neg-1)
        @inbounds a_p = a_grid[a_p_i]
        for e_3_p_i = 1:e_3_size, e_2_p_i = 1:e_2_size, e_1_p_i = 1:e_1_size
            @inbounds R[a_p_i, e_1_i, e_2_i] +=
                e_1_Γ[e_1_i, e_1_p_i] *
                e_2_Γ[e_2_i, e_2_p_i] *
                e_3_Γ[e_3_p_i] *
                (policy_d[a_p_i, e_1_p_i, e_2_p_i, e_3_p_i] * η * w * exp(e_1_grid[e_1_p_i] + e_2_grid[e_2_p_i] + e_3_grid[e_3_p_i]) + (1.0 - policy_d[a_p_i, e_1_p_i, e_2_p_i, e_3_p_i]) * (-a_p))
        end
        @inbounds q[a_p_i, e_1_i, e_2_i] = ρ * R[a_p_i, e_1_i, e_2_i] / ((-a_p) * (1.0 + τ + ι))
    end

    # return results
    return R, q
end

function solve_value_and_pricing_function!(variables::Mutable_Variables, parameters::NamedTuple; tol::Real = 1E-8, iter_max::Integer = 1000, slow_updating::Real = 1.0)
    """
    solve household and banking problems using one-loop algorithm
    """

    # initialize the iteration number and criterion
    search_iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving household and banking problems (one-loop): ")

    # construct containers
    V_p = similar(variables.V)
    V_d_p = similar(variables.V_d)
    V_nd_p = similar(variables.V_nd)
    V_pos_p = similar(variables.V_pos)
    q_p = similar(variables.q)

    while crit > tol && search_iter < iter_max

        # copy previous values
        copyto!(V_p, variables.V)
        copyto!(V_d_p, variables.V_d)
        copyto!(V_nd_p, variables.V_nd)
        copyto!(V_pos_p, variables.V_pos)
        copyto!(q_p, variables.q)

        # value and policy functions
        variables.V, variables.V_d, variables.V_nd, variables.V_pos, variables.policy_a, variables.policy_d, variables.policy_pos_a =
            value_and_policy_function(V_p, V_d_p, V_nd_p, V_pos_p, variables.q, variables.aggregate_prices.w_λ, parameters; slow_updating = slow_updating)

        # pricing function and borrowing risky limit
        variables.R, variables.q = pricing_and_rbl_function(variables.policy_d, variables.aggregate_prices.w_λ, variables.aggregate_prices.ι_λ, parameters)

        # check convergence
        V_crit = norm(variables.V .- V_p, Inf)
        V_pos_crit = norm(variables.V_pos .- V_pos_p, Inf)
        q_crit = norm(variables.q .- q_p, Inf)
        crit = max(V_crit, V_pos_crit, q_crit)

        # update the iteration number
        search_iter += 1

        # manually report convergence progress
        # println("|V| = $V_crit, |V_pos| = $V_pos_crit, |q| = $q_crit")
        # println("Solving household and banking problems (one-loop): search_iter = $search_iter and crit = $crit > tol = $tol")
        ProgressMeter.update!(prog, crit)
    end
end

# function stationary_distribution_function(μ_p::Array{Float64,5}, policy_a::Array{Float64,5}, policy_d::Array{Float64,4}, policy_pos_a::Array{Float64,5}, parameters::NamedTuple)
#     """
#     update stationary distribution
#     """
#
#     # unpack parameters
#     @unpack e_1_size, e_1_Γ, G_e_1, e_2_size, e_2_Γ, G_e_2, e_3_size, e_3_Γ, G_e_3, a_size, a_ind_zero, ρ, p_h = parameters
#
#     # construct container
#     μ = zeros(a_size, e_1_size, e_2_size, e_3_size, 2)
#
#     for e_3_i = 1:e_3_size, e_2_i = 1:e_2_size, e_1_i = 1:e_1_size, a_i = 1:a_size
#         for e_3_p_i = 1:e_3_size, e_2_p_i = 1:e_2_size, e_1_p_i = 1:e_1_size
#
#             # good credit history
#             @inbounds @views μ[:, e_1_p_i, e_2_p_i, e_3_p_i, 1] +=
#                 ρ * (1.0 - policy_d[a_i, e_1_i, e_2_i, e_3_i]) * policy_a[:, a_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 1]
#             @inbounds μ[a_ind_zero, e_1_p_i, e_2_p_i, e_3_p_i, 1] += ρ * policy_d[a_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 1]
#             @inbounds μ[a_ind_zero, e_1_p_i, e_2_p_i, e_3_p_i, 1] += (1.0 - ρ) * G_e_1[e_1_p_i] * G_e_2[e_2_p_i] * (e_3_p_i == ((e_3_size + 1) / 2)) * μ_p[a_i, e_1_i, e_2_i, e_3_i, 1]
#
#             # bad credit history
#             if a_i >= a_ind_zero
#                 a_pos_i = a_i - a_ind_zero + 1
#                 @inbounds @views μ[a_ind_zero:end, e_1_p_i, e_2_p_i, e_3_p_i, 1] +=
#                     ρ * p_h * policy_pos_a[:, a_pos_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 2]
#                 @inbounds @views μ[a_ind_zero:end, e_1_p_i, e_2_p_i, e_3_p_i, 2] +=
#                     ρ * (1.0 - p_h) * policy_pos_a[:, a_pos_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 2]
#                 @inbounds μ[a_ind_zero, e_1_p_i, e_2_p_i, e_3_p_i, 1] += (1.0 - ρ) * G_e_1[e_1_p_i] * G_e_2[e_2_p_i] * (e_3_p_i == ((e_3_size + 1) / 2)) * μ_p[a_i, e_1_i, e_2_i, e_3_i, 2]
#             end
#         end
#     end
#
#     # standardize distribution
#     μ = μ ./ sum(μ)
#
#     # return result
#     return μ
# end

function stationary_distribution_function(μ_p::Array{Float64,5}, policy_a::Array{Float64,5}, policy_d::Array{Float64,4}, policy_pos_a::Array{Float64,5}, parameters::NamedTuple)
    """
    update stationary distribution
    """

    # unpack parameters
    @unpack e_1_size, e_1_Γ, G_e_1, e_2_size, e_2_Γ, G_e_2, e_3_size, e_3_Γ, G_e_3, a_size, a_ind_zero, ρ, p_h = parameters

    # construct container
    μ = zeros(a_size, e_1_size, e_2_size, e_3_size, 2)

    for e_3_p_i = 1:e_3_size, e_2_p_i = 1:e_2_size, e_1_p_i = 1:e_1_size, a_p_i = 1:a_size
        if a_p_i < a_ind_zero
            μ_temp_1_1 = 0.0
            for e_3_i = 1:e_3_size, e_2_i = 1:e_2_size, e_1_i = 1:e_1_size, a_i = 1:a_size
                μ_temp_1_1 += ρ * (1.0 - policy_d[a_i, e_1_i, e_2_i, e_3_i]) * policy_a[a_p_i, a_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 1]
            end
            μ[a_p_i, e_1_p_i, e_2_p_i, e_3_p_i, 1] = μ_temp_1_1
        elseif a_p_i == a_ind_zero
            μ_temp_1_1 = 0.0
            μ_temp_1_2 = 0.0
            μ_temp_2_1 = 0.0
            μ_temp_2_2 = 0.0
            for e_3_i = 1:e_3_size, e_2_i = 1:e_2_size, e_1_i = 1:e_1_size, a_i = 1:a_size
                μ_temp_1_1 += ρ * (1.0 - policy_d[a_i, e_1_i, e_2_i, e_3_i]) * policy_a[a_p_i, a_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 1]
                μ_temp_1_1 += (1.0 - ρ) * G_e_1[e_1_p_i] * G_e_2[e_2_p_i] * (e_3_p_i == ((e_3_size + 1) / 2)) * μ_p[a_i, e_1_i, e_2_i, e_3_i, 1]
                μ_temp_1_2 += ρ * policy_d[a_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 1]
                if a_i >= a_ind_zero
                    a_pos_i = a_i - a_ind_zero + 1
                    μ_temp_2_1 += ρ * p_h * policy_pos_a[1, a_pos_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 2]
                    μ_temp_2_1 += (1.0 - ρ) * G_e_1[e_1_p_i] * G_e_2[e_2_p_i] * (e_3_p_i == ((e_3_size + 1) / 2)) * μ_p[a_i, e_1_i, e_2_i, e_3_i, 2]
                    μ_temp_2_2 += ρ * (1.0 - p_h) * policy_pos_a[1, a_pos_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 2]
                end
            end
            μ[a_p_i, e_1_p_i, e_2_p_i, e_3_p_i, 1] = μ_temp_1_1 + μ_temp_2_1
            μ[a_p_i, e_1_p_i, e_2_p_i, e_3_p_i, 2] = μ_temp_1_2 + μ_temp_2_2
        else
            μ_temp_1_1 = 0.0
            μ_temp_2_1 = 0.0
            μ_temp_2_2 = 0.0
            for e_3_i = 1:e_3_size, e_2_i = 1:e_2_size, e_1_i = 1:e_1_size, a_i = 1:a_size
                μ_temp_1_1 += ρ * (1.0 - policy_d[a_i, e_1_i, e_2_i, e_3_i]) * policy_a[a_p_i, a_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 1]
                if a_i >= a_ind_zero
                    a_pos_i = a_i - a_ind_zero + 1
                    a_pos_p_i = a_p_i - a_ind_zero + 1
                    μ_temp_2_1 += ρ * p_h * policy_pos_a[a_pos_p_i, a_pos_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 2]
                    μ_temp_2_2 += ρ * (1.0 - p_h) * policy_pos_a[a_pos_p_i, a_pos_i, e_1_i, e_2_i, e_3_i] * e_1_Γ[e_1_i, e_1_p_i] * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * μ_p[a_i, e_1_i, e_2_i, e_3_i, 2]
                end
            end
            μ[a_p_i, e_1_p_i, e_2_p_i, e_3_p_i, 1] = μ_temp_1_1 + μ_temp_2_1
            μ[a_p_i, e_1_p_i, e_2_p_i, e_3_p_i, 2] = μ_temp_2_2
        end
    end

    # standardize distribution
    μ = μ ./ sum(μ)

    # return result
    return μ
end

function solve_stationary_distribution_function!(variables::Mutable_Variables, parameters::NamedTuple; tol::Real = 1E-8, iter_max::Integer = 2000)
    """
    solve stationary distribution
    """

    # initialize the iteration number and criterion
    search_iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving stationary distribution: ")

    # construct container
    μ_p = similar(variables.μ)

    while crit > tol && search_iter < iter_max

        # copy previous value
        copyto!(μ_p, variables.μ)

        # update stationary distribution
        variables.μ = stationary_distribution_function(μ_p, variables.policy_a, variables.policy_d, variables.policy_pos_a, parameters)

        # check convergence
        crit = norm(variables.μ .- μ_p, Inf)

        # update the iteration number
        search_iter += 1

        # manually report convergence progress
        # println("Solving stationary distribution: search_iter = $search_iter and crit = $crit > tol = $tol")
        ProgressMeter.update!(prog, crit)
    end
end

function solve_aggregate_variable_function(policy_a::Array{Float64,5}, policy_d::Array{Float64,4}, policy_pos_a::Array{Float64,5}, q::Array{Float64,3}, μ::Array{Float64,5}, K::Real, w::Real, parameters::NamedTuple)
    """
    compute equlibrium aggregate variables
    """

    # unpack parameters
    @unpack e_1_size, e_1_grid, e_2_size, e_2_grid, e_3_size, e_3_grid, a_size, a_grid, a_ind_zero = parameters

    # initialize container
    L = 0.0
    D = 0.0
    N = 0.0
    leverage_ratio = 0.0
    KL_to_D_ratio = 0.0
    debt_to_earning_ratio = 0.0
    debt_to_earning_ratio_num = 0.0
    debt_to_earning_ratio_den = 0.0
    share_of_filers = 0.0
    share_in_debts = 0.0
    avg_loan_rate = 0.0
    avg_loan_rate_num = 0.0
    avg_loan_rate_den = 0.0
    avg_loan_rate_pw = 0.0
    avg_loan_rate_pw_num = 0.0
    avg_loan_rate_pw_den = 0.0

    # total loans, deposits, share of filers, nad debt-to-earning ratio
    for e_3_i = 1:e_3_size, e_2_i = 1:e_2_size, e_1_i = 1:e_1_size

        # interpolated discounted borrowing amount
        @inbounds @views q_e = q[:, e_1_i, e_2_i]
        qa_e = q_e .* a_grid

        # loop over the dimension of asset holding
        for a_i = 1:a_size

            # total loans
            @inbounds @views L += -μ[a_i, e_1_i, e_2_i, e_3_i, 1] * (1.0 - policy_d[a_i, e_1_i, e_2_i, e_3_i]) * sum(policy_a[1:(a_ind_zero-1), a_i, e_1_i, e_2_i, e_3_i] .* qa_e[1:(a_ind_zero-1)])

            # average loan rate
            @inbounds @views avg_loan_rate_num += μ[a_i, e_1_i, e_2_i, e_3_i, 1] * (1.0 - policy_d[a_i, e_1_i, e_2_i, e_3_i]) * sum(policy_a[1:(a_ind_zero-1), a_i, e_1_i, e_2_i, e_3_i] .* (1.0 ./ q_e[1:(a_ind_zero-1)] .- 1.0))
            @inbounds @views avg_loan_rate_den += μ[a_i, e_1_i, e_2_i, e_3_i, 1] * (1.0 - policy_d[a_i, e_1_i, e_2_i, e_3_i]) * sum(policy_a[1:(a_ind_zero-1), a_i, e_1_i, e_2_i, e_3_i])

            # average loan rate (persons-weighted)
            @inbounds @views avg_loan_rate_pw_num += (1.0 - policy_d[a_i, e_1_i, e_2_i, e_3_i]) * sum(policy_a[1:(a_ind_zero-1), a_i, e_1_i, e_2_i, e_3_i] .* (1.0 ./ q_e[1:(a_ind_zero-1)] .- 1.0))
            @inbounds @views avg_loan_rate_pw_den += (1.0 - policy_d[a_i, e_1_i, e_2_i, e_3_i]) * sum(policy_a[1:(a_ind_zero-1), a_i, e_1_i, e_2_i, e_3_i])

            # total deposits
            @inbounds @views D += μ[a_i, e_1_i, e_2_i, e_3_i, 1] * (1.0 - policy_d[a_i, e_1_i, e_2_i, e_3_i]) * sum(policy_a[(a_ind_zero+1):end, a_i, e_1_i, e_2_i, e_3_i] .* qa_e[(a_ind_zero+1):end])
            if a_i >= a_ind_zero
                a_pos_i = a_i - a_ind_zero + 1
                @inbounds @views D += μ[a_i, e_1_i, e_2_i, e_3_i, 2] * sum(policy_pos_a[2:end, a_pos_i, e_1_i, e_2_i, e_3_i] .* qa_e[(a_ind_zero+1):end])
            end

            # share of filers
            @inbounds share_of_filers += μ[a_i, e_1_i, e_2_i, e_3_i, 1] * policy_d[a_i, e_1_i, e_2_i, e_3_i]

            # debt-to-earning ratio
            if a_i < a_ind_zero
                @inbounds debt_to_earning_ratio_num += μ[a_i, e_1_i, e_2_i, e_3_i, 1] * -a_grid[a_i]
                @inbounds debt_to_earning_ratio_den += μ[a_i, e_1_i, e_2_i, e_3_i, 1] * (w * exp(e_1_grid[e_1_i] + e_2_grid[e_2_i] + e_3_grid[e_3_i]))
            end
        end
    end

    # debt-to-earning ratio
    # debt_to_earning_ratio = debt_to_earning_ratio_num / debt_to_earning_ratio_den
    debt_to_earning_ratio = L / w

    # average loan rate
    avg_loan_rate = avg_loan_rate_num / avg_loan_rate_den
    avg_loan_rate_pw = avg_loan_rate_pw_num / avg_loan_rate_pw_den

    # net worth
    N = (K + L) - D

    # leverage ratio
    leverage_ratio = (K + L) / N

    # capital-loan-to-deposit ratio
    KL_to_D_ratio = (K + L) / D

    # share in debt
    share_in_debts = sum(μ[1:(a_ind_zero-1), :, :, :, 1])

    # return results
    aggregate_variables = Mutable_Aggregate_Variables(K, L, D, N, leverage_ratio, KL_to_D_ratio, debt_to_earning_ratio, share_of_filers, share_in_debts, avg_loan_rate, avg_loan_rate_pw)
    return aggregate_variables
end

function solve_economy_function!(variables::Mutable_Variables, parameters::NamedTuple; tol_h::Real = 1E-6, tol_μ::Real = 1E-8)
    """
    solve the economy with given liquidity multiplier ι
    """

    # solve household and banking problems
    solve_value_and_pricing_function!(variables, parameters; tol = tol_h, iter_max = 500, slow_updating = 1.0)

    # solve the cross-sectional distribution
    solve_stationary_distribution_function!(variables, parameters; tol = tol_μ, iter_max = 1000)

    # compute aggregate variables
    variables.aggregate_variables =
        solve_aggregate_variable_function(variables.policy_a, variables.policy_d, variables.policy_pos_a, variables.q, variables.μ, variables.aggregate_prices.K_λ, variables.aggregate_prices.w_λ, parameters)

    # compute the difference between demand and supply sides
    ED = variables.aggregate_variables.KL_to_D_ratio - variables.aggregate_prices.KL_to_D_ratio_λ

    # printout results
    data_spec = Any[
        "Wage Garnishment Rate" parameters.η #=1=#
        "Liquidity Multiplier" variables.aggregate_prices.λ #=2=#
        "Asset-to-Debt Ratio (Demand)" variables.aggregate_variables.KL_to_D_ratio #=3=#
        "Asset-to-Debt Ratio (Supply)" variables.aggregate_prices.KL_to_D_ratio_λ #=4=#
        "Difference" ED #=5=#
    ]
    pretty_table(data_spec; header = ["Name", "Value"], alignment = [:l, :r], formatters = ft_round(8), body_hlines = [2, 4])

    # return excess demand
    return ED
end

function optimal_multiplier_function(parameters::NamedTuple; λ_min_adhoc::Real = -Inf, λ_max_adhoc::Real = Inf, tol::Real = 1E-6, iter_max::Real = 500)
    """
    solve for optimal liquidity multiplier
    """

    # check the case of λ_min = 0.0
    λ_lower = max(λ_min_adhoc, 0.0)
    variables_λ_lower = variables_function(parameters; λ = λ_lower)
    ED_λ_lower = solve_economy_function!(variables_λ_lower, parameters)
    if ED_λ_lower > 0.0
        return variables_λ_lower, variables_λ_lower, 1
    end

    # check the case of λ_max = 1-ψ^(1/2)
    λ_max = 1.0 - sqrt(parameters.ψ)
    λ_upper = min(λ_max_adhoc, λ_max)
    variables_λ_upper = variables_function(parameters; λ = λ_upper)
    ED_λ_upper = solve_economy_function!(variables_λ_upper, parameters)
    if ED_λ_upper < 0.0
        return variables_λ_lower, variables_λ_upper, 2 # meaning solution doesn't exist!
    end

    # initialization
    search_iter = 0
    crit = Inf
    λ_optimal = 0.0
    variables_λ_optimal = []

    # solve equlibrium multiplier by bisection
    while crit > tol && search_iter < iter_max

        # update the multiplier
        λ_optimal = (λ_lower + λ_upper) / 2

        # compute the associated results
        variables_λ_optimal = variables_function(parameters; λ = λ_optimal)
        ED_λ_optimal = solve_economy_function!(variables_λ_optimal, parameters)

        # update search region
        if ED_λ_optimal > 0.0
            λ_upper = λ_optimal
        else
            λ_lower = λ_optimal
        end

        # check convergence
        crit = abs(ED_λ_optimal)

        # update the iteration number
        search_iter += 1

    end

    # return results
    return variables_λ_lower, variables_λ_optimal, 3
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
    @unpack a_size, a_size_pos, e_1_size, e_2_size, e_3_size = parameters

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
    results_V_NFF = zeros(a_size, e_1_size, e_2_size, e_3_size, η_size)
    results_V_pos_NFF = zeros(a_size_pos, e_1_size, e_2_size, e_3_size, η_size)
    results_μ_NFF = zeros(a_size, e_1_size, e_2_size, e_3_size, 2, η_size)
    results_A_FF = zeros(var_size, η_size)
    results_V_FF = zeros(a_size, e_1_size, e_2_size, e_3_size, η_size)
    results_V_pos_FF = zeros(a_size_pos, e_1_size, e_2_size, e_3_size, η_size)
    results_μ_FF = zeros(a_size, e_1_size, e_2_size, e_3_size, 2, η_size)

    # compute the optimal multipliers with different η
    for η_i = 1:η_size
        parameters_η = parameters_function(η = η_grid[η_i])
        variables_NFF, variables_FF, flag = optimal_multiplier_function(parameters_η)

        # save results
        results_A_NFF[1, η_i] = parameters_η.η
        results_A_NFF[2, η_i] = variables_NFF.aggregate_prices.r_k_λ
        results_A_NFF[3, η_i] = variables_NFF.aggregate_prices.λ
        results_A_NFF[4, η_i] = variables_NFF.aggregate_prices.ι_λ
        results_A_NFF[5, η_i] = variables_NFF.aggregate_variables.K
        results_A_NFF[6, η_i] = variables_NFF.aggregate_variables.L
        results_A_NFF[7, η_i] = variables_NFF.aggregate_variables.D
        results_A_NFF[8, η_i] = variables_NFF.aggregate_variables.N
        results_A_NFF[9, η_i] = variables_NFF.aggregate_variables.leverage_ratio
        results_A_NFF[10, η_i] = variables_NFF.aggregate_variables.share_of_filers
        results_A_NFF[11, η_i] = variables_NFF.aggregate_variables.share_in_debts
        results_A_NFF[12, η_i] = variables_NFF.aggregate_variables.debt_to_earning_ratio
        results_A_NFF[13, η_i] = variables_NFF.aggregate_variables.avg_loan_rate_pw
        results_V_NFF[:, :, :, :, η_i] = variables_NFF.V
        results_V_pos_NFF[:, :, :, :, η_i] = variables_NFF.V_pos
        results_μ_NFF[:, :, :, :, :, η_i] = variables_NFF.μ

        results_A_FF[1, η_i] = parameters_η.η
        results_A_FF[2, η_i] = variables_FF.aggregate_prices.r_k_λ
        results_A_FF[3, η_i] = variables_FF.aggregate_prices.λ
        results_A_FF[4, η_i] = variables_FF.aggregate_prices.ι_λ
        results_A_FF[5, η_i] = variables_FF.aggregate_variables.K
        results_A_FF[6, η_i] = variables_FF.aggregate_variables.L
        results_A_FF[7, η_i] = variables_FF.aggregate_variables.D
        results_A_FF[8, η_i] = variables_FF.aggregate_variables.N
        results_A_FF[9, η_i] = variables_FF.aggregate_variables.leverage_ratio
        results_A_FF[10, η_i] = variables_FF.aggregate_variables.share_of_filers
        results_A_FF[11, η_i] = variables_FF.aggregate_variables.share_in_debts
        results_A_FF[12, η_i] = variables_FF.aggregate_variables.debt_to_earning_ratio
        results_A_FF[13, η_i] = variables_FF.aggregate_variables.avg_loan_rate_pw
        results_V_FF[:, :, :, :, η_i] = variables_FF.V
        results_V_pos_FF[:, :, :, :, η_i] = variables_FF.V_pos
        results_μ_FF[:, :, :, :, :, η_i] = variables_FF.μ
    end

    # return results
    return parameters, var_names, results_A_NFF, results_V_NFF, results_V_pos_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_V_pos_FF, results_μ_FF
end

function results_CEV_function(parameters::NamedTuple, results_V::Array{Float64,5}, results_V_pos::Array{Float64,5})
    """
    compute consumption equivalent variation (CEV) with various η compared to the smallest η (most lenient policy)
    """

    @unpack a_grid, a_size, a_grid_pos, a_size_pos, a_ind_zero, e_1_size, e_2_size, e_3_size, σ = parameters

    # initialize result matrix
    η_size = size(results_V)[end]
    results_CEV = zeros(a_size, e_1_size, e_2_size, e_3_size, 2, η_size)

    # compute CEV for different η compared to the smallest η
    for η_i = 1:η_size, e_1_i = 1:e_1_size, e_2_i = 1:e_2_size, e_3_i = 1:e_3_size, a_i = 1:a_size
        @inbounds V_new = results_V[a_i, e_1_i, e_2_i, e_3_i, η_i]
        @inbounds V_old = results_V[a_i, e_1_i, e_2_i, e_3_i, end]
        @inbounds results_CEV[a_i, e_1_i, e_2_i, e_3_i, 1, η_i] = (V_new / V_old) ^ (1.0 / (1.0 - σ)) - 1.0
        if a_i >= a_ind_zero
            a_pos_i = a_i - a_ind_zero + 1
            @inbounds V_pos_new = results_V_pos[a_pos_i, e_1_i, e_2_i, e_3_i, η_i]
            @inbounds V_pos_old = results_V_pos[a_pos_i, e_1_i, e_2_i, e_3_i, end]
            @inbounds results_CEV[a_pos_i, e_1_i, e_2_i, e_3_i, 2, η_i] = (V_pos_new / V_pos_old) ^ (1.0 / (1.0 - σ)) - 1.0
        end
    end

    # return results
    return results_CEV
end

#=================#
# Solve the model #
#=================#
parameters = parameters_function()
# variables = variables_function(parameters; λ = 0.0)
# solve_economy_function!(variables, parameters)

variables_min = variables_function(parameters; λ = 0.0)
solve_economy_function!(variables_min, parameters)
flag = 1

variables_max = variables_function(parameters; λ = 1.0 - sqrt(parameters.ψ))
solve_economy_function!(variables_max, parameters)
flag = 2

compare_results = [
    variables_min.aggregate_prices.λ variables_max.aggregate_prices.λ
    variables_min.aggregate_prices.KL_to_D_ratio_λ variables_max.aggregate_prices.KL_to_D_ratio_λ
    variables_min.aggregate_variables.KL_to_D_ratio variables_max.aggregate_variables.KL_to_D_ratio
    variables_min.aggregate_variables.share_of_filers*100 variables_max.aggregate_variables.share_of_filers*100
    variables_min.aggregate_variables.share_in_debts*100 variables_max.aggregate_variables.share_in_debts*100
    variables_min.aggregate_variables.debt_to_earning_ratio*100 variables_max.aggregate_variables.debt_to_earning_ratio*100
    variables_min.aggregate_variables.avg_loan_rate*100 variables_max.aggregate_variables.avg_loan_rate*100
    # variables_min.policy_a[end,end,end,end,2] < parameters.a_grid[end]  variables_max.policy_a[end,end,end,end,2] < parameters.a_grid[end]
]

variables = variables_function(parameters; λ = 0.0381861)
solve_economy_function!(variables, parameters)
flag = 3

calibration_results = [
    parameters.β,
    parameters.δ,
    parameters.τ,
    parameters.p_h,
    parameters.η,
    parameters.ψ,
    parameters.θ,
    parameters.ζ_a,
    parameters.ζ_d,
    variables.aggregate_prices.λ,
    variables.aggregate_variables.KL_to_D_ratio,
    variables.aggregate_variables.share_of_filers * 100,
    variables.aggregate_variables.D / variables.aggregate_variables.L,
    variables.aggregate_variables.share_in_debts * 100,
    variables.aggregate_variables.debt_to_earning_ratio * 100,
    variables.aggregate_variables.avg_loan_rate * 100,
    flag
    ]

# parameters = parameters_function()
# variables = variables_function(parameters; λ = 0.02496311756496223)
# solve_economy_function!(variables, parameters)

#=============#
# Calibration #
#=============#
β_search = 0.93 / 0.98 # collect(0.94:0.01:0.97)
θ_search = 1.0 / 3.0 # eps() # collect(0.04:0.001:0.07)
η_search = 0.25 # collect(0.20:0.05:0.40)
ζ_a_search = 0.005 # collect(0.004:0.001:0.006)
ζ_d_search = 0.18 # collect(0.16:0.005:0.17)
β_search_szie = length(β_search)
θ_search_szie = length(θ_search)
η_search_szie = length(η_search)
ζ_a_search_szie = length(ζ_a_search)
ζ_d_search_szie = length(ζ_d_search)
search_size = β_search_szie * θ_search_szie * η_search_szie * ζ_a_search_szie * ζ_d_search_szie
calibration_results = zeros(search_size, 17)

for β_i in 1:β_search_szie, θ_i in 1:θ_search_szie, η_i in 1:η_search_szie, ζ_a_i in 1:ζ_a_search_szie, ζ_d_i in 1:ζ_d_search_szie
    parameters = parameters_function(β = β_search[β_i], θ = θ_search[θ_i], η = η_search[η_i], ζ_a = ζ_a_search[ζ_a_i], ζ_d = ζ_d_search[ζ_d_i])
    # variables = variables_function(parameters; λ = 0.0)
    # solve_economy_function!(variables, parameters)
    # flag = 1
    variables_λ_lower, variables, flag = optimal_multiplier_function(parameters)

    search_iter = (β_i-1)*(θ_search_szie*η_search_szie*ζ_a_search_szie*ζ_d_search_szie) + (θ_i-1)*(η_search_szie*ζ_a_search_szie*ζ_d_search_szie) + (η_i-1)*ζ_a_search_szie*ζ_d_search_szie + (ζ_a_i-1)*ζ_d_search_szie + ζ_d_i
    calibration_results[search_iter, 1] = parameters.β
    calibration_results[search_iter, 2] = parameters.δ
    calibration_results[search_iter, 3] = parameters.τ
    calibration_results[search_iter, 4] = parameters.p_h
    calibration_results[search_iter, 5] = parameters.η
    calibration_results[search_iter, 6] = parameters.ψ
    calibration_results[search_iter, 7] = parameters.θ
    calibration_results[search_iter, 8] = parameters.ζ_a
    calibration_results[search_iter, 9] = parameters.ζ_d
    calibration_results[search_iter, 10] = variables.aggregate_prices.λ
    calibration_results[search_iter, 11] = variables.aggregate_variables.KL_to_D_ratio
    calibration_results[search_iter, 12] = variables.aggregate_variables.share_of_filers * 100
    calibration_results[search_iter, 13] = variables.aggregate_variables.D / variables.aggregate_variables.L
    calibration_results[search_iter, 14] = variables.aggregate_variables.share_in_debts * 100
    calibration_results[search_iter, 15] = variables.aggregate_variables.debt_to_earning_ratio * 100
    calibration_results[search_iter, 16] = variables.aggregate_variables.avg_loan_rate * 100
    calibration_results[search_iter, 17] = flag
end

# cd(homedir() * "/financial_frictions/")
cd(homedir() * "\\Dropbox\\Dissertation\\Chapter 3 - Consumer Bankruptcy with Financial Frictions\\")
CSV.write("calibration_julia.csv", Tables.table(calibration_results), writeheader=false)

# short_results = [
#     calibration_results[1,9]
#     calibration_results[1,10]
#     calibration_results[1,10]
#     calibration_results[1,11]
#     calibration_results[1,13]
#     calibration_results[1,14]
#     calibration_results[1,15]
#     -Inf
# ]

#======================================================#
# Solve the model with different bankruptcy strictness #
#======================================================#
# var_names, results_A_NFF, results_V_NFF, results_V_pos_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_V_pos_FF, results_μ_FF = results_η_function(η_min = 0.10, η_max = 0.90, η_step = 0.10)
# cd(homedir() * "/financial_frictions/")
cd(homedir() * "\\Dropbox\\Dissertation\\Chapter 3 - Consumer Bankruptcy with Financial Frictions\\")
# @save "results_eta.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF
@load "results_eta.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

#=============================#
# Solve transitional dynamics #
#=============================#
# mutable struct Mutable_Aggregate_Prices_T
#     """
#     construct a type for mutable aggregate prices of periods T
#     """
#     λ::Array{Float64,1}
#     ξ::Array{Float64,1}
#     Λ::Array{Float64,1}
#     leverage_ratio_λ::Array{Float64,1}
#     KL_to_D_ratio_λ::Array{Float64,1}
#     ι::Array{Float64,1}
#     r_k::Array{Float64,1}
#     K_λ::Array{Float64,1}
#     w::Array{Float64,1}
# end
#
# mutable struct Mutable_Aggregate_Variables_T
#     """
#     construct a type for mutable aggregate variables of periods T
#     """
#     K::Array{Float64,1}
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
#
# mutable struct Mutable_Variables_T
#     """
#     construct a type for mutable variables of periods T
#     """
#     aggregate_prices::Mutable_Aggregate_Prices_T
#     aggregate_variables::Mutable_Aggregate_Variables_T
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
# end
#
# function variables_T_function(variables_old::Mutable_Variables, variables_new::Mutable_Variables, parameters_new::NamedTuple; T_size::Integer)
#     """
#     construct a mutable object containing endogenous variables of periods T
#     """
#
#     # unapcl parameters from new steady state
#     @unpack a_size, a_size_neg, a_size_μ, e_size, t_size, ν_size = parameters_new
#     @unpack θ, β_f, ψ, r_f, E, δ, α = parameters_new
#
#     # adjust periods considered
#     T_size = T_size + 2
#
#     # leverage_ratio_λ = collect(range(variables_new.aggregate_prices.leverage_ratio_λ, variables_new.aggregate_prices.leverage_ratio_λ, length = T_size))
#     # leverage_ratio_λ = collect(range(variables_old.aggregate_variables.leverage_ratio, variables_new.aggregate_variables.leverage_ratio, length = T_size))
#     leverage_ratio_λ = collect(range(variables_new.aggregate_variables.leverage_ratio, variables_new.aggregate_variables.leverage_ratio, length = T_size))
#     ξ = θ .* leverage_ratio_λ
#     Λ = β_f .* (1.0 .- ψ .+ ψ .* ξ)
#     KL_to_D_ratio_λ = leverage_ratio_λ ./ (leverage_ratio_λ .- 1.0)
#     λ = zeros(T_size)
#     λ[1] = 1.0 - ((1.0 - ψ + ψ*ξ[1]) / ξ[1])
#     λ[end] = 1.0 - ((1.0 - ψ + ψ*ξ[end]) / ξ[end])
#     ι = zeros(T_size)
#     ι[1] = λ[1] * θ / Λ[1]
#     ι[end] = λ[end] * θ / Λ[end]
#     r_k = zeros(T_size)
#     r_k[1] = variables_old.aggregate_prices.r_k
#     r_k[end] = variables_new.aggregate_prices.r_k
#     K_λ = zeros(T_size)
#     K_λ[1] = variables_old.aggregate_prices.K_λ
#     K_λ[end] = variables_new.aggregate_prices.K_λ
#     w  = zeros(T_size)
#     w[1] = variables_old.aggregate_prices.w
#     w[end] = variables_new.aggregate_prices.w
#     for T_i = (T_size-1):(-1):2
#         λ[T_i] = 1.0 - ((1.0 - ψ + ψ*ξ[T_i+1]) / ξ[T_i])
#         ι[T_i] = λ[T_i] * θ / Λ[T_i+1]
#         r_k[T_i] = r_f + ι[T_i]
#         K_λ[T_i] = E .* ((r_k[T_i] .+ δ) ./ α).^(1.0 / (α - 1.0))
#         w[T_i] = (1.0 - α) .* ((K_λ[T_i] ./ E).^α)
#     end
#     aggregate_prices = Mutable_Aggregate_Prices_T(λ, ξ, Λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι, r_k, K_λ, w)
#
#     # define aggregate variables
#     K = zeros(T_size)
#     K[1] = K_λ[1]
#     K[end] = K_λ[end]
#     L = zeros(T_size)
#     L[1] = variables_old.aggregate_variables.L
#     L[end] = variables_new.aggregate_variables.L
#     D = zeros(T_size)
#     D[1] = variables_old.aggregate_variables.D
#     D[end] = variables_new.aggregate_variables.D
#     N = zeros(T_size)
#     N[1] = variables_old.aggregate_variables.N
#     N[end] = variables_new.aggregate_variables.N
#     leverage_ratio = zeros(T_size)
#     leverage_ratio[1] = variables_old.aggregate_variables.leverage_ratio
#     leverage_ratio[end] = variables_new.aggregate_variables.leverage_ratio
#     KL_to_D_ratio = zeros(T_size)
#     KL_to_D_ratio[1] = variables_old.aggregate_variables.KL_to_D_ratio
#     KL_to_D_ratio[end] = variables_new.aggregate_variables.KL_to_D_ratio
#     debt_to_earning_ratio = zeros(T_size)
#     debt_to_earning_ratio[1] = variables_old.aggregate_variables.debt_to_earning_ratio
#     debt_to_earning_ratio[end] = variables_new.aggregate_variables.debt_to_earning_ratio
#     share_of_filers = zeros(T_size)
#     share_of_filers[1] = variables_old.aggregate_variables.share_of_filers
#     share_of_filers[end] = variables_new.aggregate_variables.share_of_filers
#     share_of_involuntary_filers = zeros(T_size)
#     share_of_involuntary_filers[1] = variables_old.aggregate_variables.share_of_involuntary_filers
#     share_of_involuntary_filers[end] = variables_new.aggregate_variables.share_of_involuntary_filers
#     share_in_debts = zeros(T_size)
#     share_in_debts[1] = variables_old.aggregate_variables.share_in_debts
#     share_in_debts[end] = variables_new.aggregate_variables.share_in_debts
#     avg_loan_rate = zeros(T_size)
#     avg_loan_rate[1] = variables_old.aggregate_variables.avg_loan_rate
#     avg_loan_rate[end] = variables_new.aggregate_variables.avg_loan_rate
#     avg_loan_rate_pw = zeros(T_size)
#     avg_loan_rate_pw[1] = variables_old.aggregate_variables.avg_loan_rate_pw
#     avg_loan_rate_pw[end] = variables_new.aggregate_variables.avg_loan_rate_pw
#     aggregate_variables = Mutable_Aggregate_Variables_T(K, L, D, N, leverage_ratio, KL_to_D_ratio, debt_to_earning_ratio, share_of_filers, share_of_involuntary_filers, share_in_debts, avg_loan_rate, avg_loan_rate_pw)
#
#     # define repayment probability, pricing function, and risky borrowing limit
#     R = zeros(a_size_neg, e_size, T_size)
#     R[:,:,1] = variables_old.R
#     R[:,:,(end-1):end] .= variables_new.R
#     q = ones(a_size, e_size, T_size) ./ (1.0 + r_f)
#     q[:,:,1] = variables_old.q
#     q[:,:,(end-1):end] .= variables_new.q
#     rbl = zeros(e_size, 2, T_size)
#     rbl[:,:,1] = variables_old.rbl
#     rbl[:,:,(end-1):end] .= variables_new.rbl
#
#     # define value and policy functions
#     V = zeros(a_size, e_size, t_size, ν_size, T_size)
#     V[:,:,:,:,1] = variables_old.V
#     V[:,:,:,:,end] = variables_new.V
#     V_d = zeros(e_size, t_size, ν_size, T_size)
#     V_d[:,:,:,1] = variables_old.V_d
#     V_d[:,:,:,end] = variables_new.V_d
#     V_nd = zeros(a_size, e_size, t_size, ν_size, T_size)
#     V_nd[:,:,:,:,1] = variables_old.V_nd
#     V_nd[:,:,:,:,end] = variables_new.V_nd
#     policy_a = zeros(a_size, e_size, t_size, ν_size, T_size)
#     policy_a[:,:,:,:,1] = variables_old.policy_a
#     policy_a[:,:,:,:,end] = variables_new.policy_a
#
#     # define thresholds conditional on endowment or asset
#     threshold_a = zeros(e_size, t_size, ν_size, T_size)
#     threshold_a[:,:,:,1] = variables_old.threshold_a
#     threshold_a[:,:,:,end] = variables_new.threshold_a
#     threshold_e = zeros(a_size, t_size, ν_size, T_size)
#     threshold_e[:,:,:,1] = variables_old.threshold_e
#     threshold_e[:,:,:,end] = variables_new.threshold_e
#
#     # define cross-sectional distribution
#     μ_size = a_size_μ * e_size * t_size * ν_size
#     μ = ones(a_size_μ, e_size, t_size, ν_size, T_size) ./ μ_size
#     μ[:,:,:,:,1] = variables_old.μ
#     μ[:,:,:,:,end] = variables_new.μ
#
#     # return outputs
#     variables_T = Mutable_Variables_T(aggregate_prices, aggregate_variables, R, q, rbl, V, V_d, V_nd, policy_a, threshold_a, threshold_e, μ)
#     return variables_T
# end
#
# function transitional_dynamic_λ_function!(variables_T::Mutable_Variables_T, parameters_new::NamedTuple; tol::Real = 1E-4, iter_max::Real = 500, slow_updating::Real = 1.0, figure_track::Bool = true)
#     """
#     solve transitional dynamics of periods T from initial to new steady states
#     """
#
#     # unpack parameters
#     @unpack θ, ψ, β_f, r_f, E, δ, α = parameters_new
#
#     # initialize the iteration number and criterion
#     search_iter = 0
#     crit = Inf
#
#     # obtain number of periods
#     T_size = length(variables_T.aggregate_prices.leverage_ratio_λ)
#
#     # construct container
#     leverage_ratio_λ_p = similar(variables_T.aggregate_prices.leverage_ratio_λ)
#
#     while crit > tol && search_iter < iter_max
#
#         # copy previous value
#         copyto!(leverage_ratio_λ_p, variables_T.aggregate_prices.leverage_ratio_λ)
#
#         # solve individual-level problems backward
#         println("Solving individual-level problems backward...")
#         for T_i = (T_size-1):(-1):2
#             # pricing function and borrowing risky limit
#             variables_T.R[:,:,T_i], variables_T.q[:,:,T_i], variables_T.rbl[:,:,T_i] = pricing_and_rbl_function(variables_T.threshold_e[:,:,:,T_i+1], variables_T.aggregate_prices.ι[T_i], parameters)
#
#             # value and policy functions
#             variables_T.V[:,:,:,:,T_i], variables_T.V_d[:,:,:,T_i], variables_T.V_nd[:,:,:,:,T_i], variables_T.policy_a[:,:,:,:,T_i] = value_and_policy_function(variables_T.V[:,:,:,:,T_i+1], variables_T.V_d[:,:,:,T_i+1], variables_T.V_nd[:,:,:,:,T_i+1], variables_T.q[:,:,T_i], variables_T.rbl[:,:,T_i], variables_T.aggregate_prices.w[T_i], parameters)
#
#             # thresholds
#             variables_T.threshold_a[:,:,:,T_i], variables_T.threshold_e[:,:,:,T_i] = threshold_function(variables_T.V_d[:,:,:,T_i], variables_T.V_nd[:,:,:,:,T_i], variables_T.aggregate_prices.w[T_i], parameters)
#         end
#
#         # solve distribution forward and update aggregate variables and prices
#         println("Solving distribution and aggregate variables/prices forward...")
#         for T_i = 2:(T_size-1)
#             # update stationary distribution
#             variables_T.μ[:,:,:,:,T_i] = stationary_distribution_function(variables_T.μ[:,:,:,:,T_i-1], variables_T.policy_a[:,:,:,:,T_i], variables_T.threshold_a[:,:,:,T_i], parameters_new)
#
#             # compute aggregate variables
#             aggregate_variables = solve_aggregate_variable_function(variables_T.policy_a[:,:,:,:,T_i], variables_T.threshold_a[:,:,:,T_i], variables_T.q[:,:,T_i], variables_T.rbl[:,:,T_i], variables_T.μ[:,:,:,:,T_i], variables_T.aggregate_prices.K_λ[T_i], variables_T.aggregate_prices.w[T_i], parameters_new)
#             variables_T.aggregate_variables.K[T_i] = aggregate_variables.K
#             variables_T.aggregate_variables.L[T_i] = aggregate_variables.L
#             variables_T.aggregate_variables.D[T_i] = aggregate_variables.D
#             variables_T.aggregate_variables.N[T_i] = aggregate_variables.N
#             variables_T.aggregate_variables.leverage_ratio[T_i] = aggregate_variables.leverage_ratio
#             variables_T.aggregate_variables.KL_to_D_ratio[T_i] = aggregate_variables.KL_to_D_ratio
#             variables_T.aggregate_variables.debt_to_earning_ratio[T_i] = aggregate_variables.debt_to_earning_ratio
#             variables_T.aggregate_variables.share_of_filers[T_i] = aggregate_variables.share_of_filers
#             variables_T.aggregate_variables.share_of_involuntary_filers[T_i] = aggregate_variables.share_of_involuntary_filers
#             variables_T.aggregate_variables.share_in_debts[T_i] = aggregate_variables.share_in_debts
#             variables_T.aggregate_variables.avg_loan_rate[T_i] = aggregate_variables.avg_loan_rate
#             variables_T.aggregate_variables.avg_loan_rate_pw[T_i] = aggregate_variables.avg_loan_rate_pw
#         end
#
#         # check convergence
#         crit = norm(variables_T.aggregate_variables.leverage_ratio .- leverage_ratio_λ_p, Inf)
#
#         # update the iteration number
#         search_iter += 1
#
#         # manually report convergence progress
#         println("Solving transitional dynamics: search_iter = $search_iter and crit = $crit > tol = $tol")
#
#         # update leverage ratio
#         variables_T.aggregate_prices.leverage_ratio_λ = slow_updating * leverage_ratio_λ_p + (1.0 - slow_updating) * variables_T.aggregate_variables.leverage_ratio
#
#         # update aggregate prices
#         variables_T.aggregate_prices.ξ = θ .* variables_T.aggregate_prices.leverage_ratio_λ
#         variables_T.aggregate_prices.Λ = β_f .* (1.0 .- ψ .+ ψ .* variables_T.aggregate_prices.ξ)
#         variables_T.aggregate_prices.KL_to_D_ratio_λ = variables_T.aggregate_prices.leverage_ratio_λ ./ (variables_T.aggregate_prices.leverage_ratio_λ .- 1.0)
#         for T_i = 2:(T_size-1)
#             variables_T.aggregate_prices.λ[T_i] = 1.0 - (1.0 - ψ + ψ*variables_T.aggregate_prices.ξ[T_i+1]) / variables_T.aggregate_prices.ξ[T_i]
#             variables_T.aggregate_prices.ι[T_i] = variables_T.aggregate_prices.λ[T_i] * θ / variables_T.aggregate_prices.Λ[T_i+1]
#         end
#         variables_T.aggregate_prices.r_k = r_f .+ variables_T.aggregate_prices.ι
#         variables_T.aggregate_prices.K_λ = E .* ((variables_T.aggregate_prices.r_k .+ δ) ./ α).^(1.0 / (α - 1.0))
#         variables_T.aggregate_prices.w = (1.0 - α) .* ((variables_T.aggregate_prices.K_λ ./ E).^α)
#
#         # tracking figures
#         if figure_track == true
#             println()
#             plt_LR = lineplot(
#                 collect(2:(T_size-1)),
#                 variables_T.aggregate_variables.leverage_ratio[2:(T_size-1)],
#                 name = "updated",
#                 title = "leverage ratio",
#                 xlim = [0.0, T_size],
#                 width = 50,
#                 height = 10,
#             )
#             lineplot!(plt_LR, collect(2:(T_size-1)), leverage_ratio_λ_p[2:(T_size-1)], name = "initial")
#             lineplot!(plt_LR, collect(2:(T_size-1)), variables_T.aggregate_prices.leverage_ratio_λ[2:(T_size-1)], name = "slow updating")
#             println(plt_LR)
#         end
#     end
# end
#
# # old stationary equilibrium
# println("Solving initial steady state...")
# parameters_old = parameters_function()
# variables_old = variables_function(parameters_old; λ = 0.0)
# solve_economy_function!(variables_old, parameters_old)
#
# # new stationary equilibrium
# println("Solving new steady state...")
# parameters_new = parameters_function()
# variables_new = variables_function(parameters_new; λ = λ_optimal)
# solve_economy_function!(variables_new, parameters_new)
#
# # solve transitional dynamics
# variables_T = variables_T_function(variables_old, variables_new, parameters_new; T_size = 240)
# transitional_dynamic_λ_function!(variables_T, parameters_new; iter_max = 4, slow_updating = 0.5)
