using Parameters
using LinearAlgebra
using ProgressMeter
using Plots
using QuantEcon: tauchen, rouwenhorst
using Distributions
using QuadGK

#==================#
# Define functions #
#==================#
function parameters_function(;
    β::Real = 0.94,                 # HHs discount factor
    r_f::Real = 0.04,               # risk-free rate
    τ::Real = 0.04,                 # transaction cost
    σ::Real = 2.00,                 # CRRA coefficient
    η::Real = 0.25,                # garnishment rate
    δ::Real = 0.08,                 # depreciation rate
    α::Real = 1.0 / 3.0,            # capital share
    ψ::Real = 0.972,                # exogenous retention ratio
    θ::Real = 0.381,                # diverting fraction
    ζ_a::Real = 0.1,               # EV scale parameter (asset choice)
    ζ_d::Real = 0.1,               # EV scale parameter (default)
    e_ρ::Real = 0.9630,             # AR(1) of persistent endowment shock
    e_σ::Real = 0.1300,             # s.d. of persistent endowment shock
    e_size::Integer = 3,            # number of persistent endowment shock
    z_σ::Real = 0.35,               # s.d. of transitory endowment shock
    z_size::Integer = 3,            # number oftransitory endowment shock
    a_min::Real = -5.0,             # min of asset holding
    a_max::Real = 30.0,             # max of asset holding
    a_size_neg::Integer = 101,      # number of grid of negative asset holding for VFI
    a_size_pos::Integer = 51,       # number of grid of positive asset holding for VFI
    a_degree::Integer = 3,          # curvature of the positive asset gridpoints
    h_size::Integer = 2,            # good and bad credit history
    p_h::Real = 1.0 / 7,            # probability of credit history erased
)
    """
    contruct an immutable object containg all paramters
    """

    # persistent endowment shock
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
    e_grid, e_Γ = adda_cooper(e_size, e_ρ, e_σ)
    e_grid = exp.(e_grid)
    # e_MC = rouwenhorst(e_size, e_ρ, e_σ, 0.0)
    # e_Γ = e_MC.p
    # e_grid = exp.(collect(e_MC.state_values))

    # transitory endowment shock
    z_bar = sqrt((z_size / (z_size - 1)) * (z_σ^2))
    z_grid = [-z_bar 0 z_bar]
    z_grid = exp.(z_grid)
    z_Γ = [1.0 / z_size for i = 1:z_size]

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length = a_size_neg))
    a_grid_pos = ((range(0.0, stop = a_size_pos - 1, length = a_size_pos) / (a_size_pos - 1)) .^ a_degree) * a_max
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero, a_grid)[]

    # return values
    return (
        β = β,
        r_f = r_f,
        τ = τ,
        σ = σ,
        η = η,
        δ = δ,
        α = α,
        ψ = ψ,
        θ = θ,
        ζ_a = ζ_a,
        ζ_d = ζ_d,
        e_ρ = e_ρ,
        e_σ = e_σ,
        e_size = e_size,
        e_Γ = e_Γ,
        e_grid = e_grid,
        z_σ = z_σ,
        z_size = z_size,
        z_Γ = z_Γ,
        z_grid = z_grid,
        a_degree = a_degree,
        a_grid = a_grid,
        a_grid_neg = a_grid_neg,
        a_grid_pos = a_grid_pos,
        a_size = a_size,
        a_size_neg = a_size_neg,
        a_size_pos = a_size_pos,
        a_ind_zero = a_ind_zero,
        p_h = p_h,
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
    share_of_involuntary_filers::Real
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
    q::Array{Float64,2}
    V_good::Array{Float64,3}
    V_good_repay_all::Array{Float64,4}
    V_good_repay::Array{Float64,3}
    V_good_default::Array{Float64,3}
    V_bad_all::Array{Float64,4}
    V_bad::Array{Float64,3}
    policy_good_a::Array{Float64,4}
    policy_good_d::Array{Float64,3}
    policy_bad_a::Array{Float64,4}
    μ::Array{Float64,4}
end

function aggregate_prices_λ_funtion(parameters::NamedTuple; λ::Real)
    """
    compute aggregate prices for given incentive multiplier λ
    """
    @unpack α, ψ, θ, r_f, δ = parameters

    ξ_λ = (1.0 - ψ) / (1 - λ - ψ)
    Λ_λ = (1 + r_f)^(-1) * (1.0 - ψ + ψ * ξ_λ)
    leverage_ratio_λ = ξ_λ / θ
    KL_to_D_ratio_λ = leverage_ratio_λ / (leverage_ratio_λ - 1.0)
    ι_λ = λ * θ / Λ_λ
    r_k_λ = r_f + ι_λ
    K_λ = (α / (r_k_λ + δ))^(1.0 / (1.0 - α))
    w_λ = (1.0 - α) * K_λ^α

    return ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ
end

function variables_function(parameters::NamedTuple; λ::Real)
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack a_size, a_size_pos, a_size_neg, e_size, z_size, r_f, τ = parameters

    # define aggregate prices and variables
    ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ = aggregate_prices_λ_funtion(parameters; λ = λ)
    K = 0.0
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
    aggregate_prices = Mutable_Aggregate_Prices(λ, ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ)
    aggregate_variables =
        Mutable_Aggregate_Variables(K, L, D, N, leverage_ratio, KL_to_D_ratio, debt_to_earning_ratio, share_of_filers, share_of_involuntary_filers, share_in_debts, avg_loan_rate, avg_loan_rate_pw)

    # pricing function
    q = ones(a_size, e_size) ./ (1.0 + r_f)
    q[1:a_size_neg, :] .= 1.0 / (1.0 + r_f + τ + ι_λ)

    # value functions
    V_good = zeros(a_size, e_size, z_size)
    V_good_repay_all = zeros(a_size, a_size, e_size, z_size)
    V_good_repay = zeros(a_size, e_size, z_size)
    V_good_default = zeros(a_size, e_size, z_size)
    V_bad_all = zeros(a_size_pos, a_size_pos, e_size, z_size)
    V_bad = zeros(a_size_pos, e_size, z_size)

    # policy functions (choice probability)
    policy_good_a = zeros(a_size, a_size, e_size, z_size)
    policy_good_d = zeros(a_size, e_size, z_size)
    policy_bad_a = zeros(a_size_pos, a_size_pos, e_size, z_size)

    # define cross-sectional distribution
    μ_size = a_size * e_size * z_size * 2
    μ = ones(a_size, e_size, z_size, 2) ./ μ_size

    # return outputs
    variables = Mutable_Variables(aggregate_prices, aggregate_variables, q, V_good, V_good_repay_all, V_good_repay, V_good_default, V_bad_all, V_bad, policy_good_a, policy_good_d, policy_bad_a, μ)
    return variables
end

function utility_function(c::Real, σ::Real)
    """
    compute utility of CRRA utility function with coefficient σ
    """

    if c > 0.0
        return σ == 1.0 ? log(c) : 1.0 / ((1.0 - σ) * c^(σ - 1.0))
    else
        return -Inf
    end
end

function value_function!(V_good::Array{Float64,3}, V_bad::Array{Float64,3}, variables::Mutable_Variables, parameters::NamedTuple)
    """
    compute value functions
    """

    @unpack a_grid, a_grid_pos, a_ind_zero, a_size, a_size_pos, e_grid, e_size, e_Γ, z_grid, z_size, z_Γ, β, σ, η, p_h, ζ_a, ζ_d = parameters

    # precompute continuation value with good credit history
    V_good_expect = zeros(a_size, e_size)
    for e_i = 1:e_size, a_p_i = 1:a_size
        for z_p_i = 1:z_size, e_p_i = 1:e_size
            V_good_expect[a_p_i, e_i] += e_Γ[e_i, e_p_i] * z_Γ[z_p_i] * V_good[a_p_i, e_p_i, z_p_i]
        end
    end

    # precompute continuation value with bad credit history
    V_bad_expect = zeros(a_size_pos, e_size)
    for e_i = 1:e_size, a_pos_p_i = 1:a_size_pos
        for z_p_i = 1:z_size, e_p_i = 1:e_size
            V_bad_expect[a_pos_p_i, e_i] += e_Γ[e_i, e_p_i] * z_Γ[z_p_i] * V_bad[a_pos_p_i, e_p_i, z_p_i]
        end
    end

    for z_i = 1:z_size, e_i = 1:e_size, a_i = 1:a_size

        # extract states
        z = z_grid[z_i]
        e = e_grid[e_i]
        a = a_grid[a_i]
        y = variables.aggregate_prices.w_λ * e * z

        # good credit history and repay
        c_repay = y .+ a .- variables.q[:, e_i] .* a_grid
        # variables.V_good_repay_all[:, a_i, e_i, z_i] = (1.0 - β) * utility_function.(c_repay, σ) .+ β * V_good_expect[:, e_i]
        variables.V_good_repay_all[:, a_i, e_i, z_i] = utility_function.(c_repay, σ) .+ β * V_good_expect[:, e_i]
        V_good_repay_max = maximum(variables.V_good_repay_all[:, a_i, e_i, z_i])
        if V_good_repay_max == -Inf
            variables.V_good_repay[a_i, e_i, z_i] = V_good_repay_max
            variables.policy_good_a[:, a_i, e_i, z_i] .= 0.0
        else
            V_good_repay_exp = exp.((variables.V_good_repay_all[:, a_i, e_i, z_i] .- V_good_repay_max) ./ ζ_a)
            V_good_repay_sum = sum(V_good_repay_exp)
            variables.V_good_repay[a_i, e_i, z_i] = V_good_repay_max + ζ_a * log(V_good_repay_sum)
            variables.policy_good_a[:, a_i, e_i, z_i] = V_good_repay_exp ./ V_good_repay_sum
        end

        # good credit history and default
        c_default = (1.0 - η) * y
        # variables.V_good_default[a_i, e_i, z_i] = a < -η*y ? (1.0 - β) * utility_function(c_default, σ) + β * V_bad_expect[1, e_i] : -Inf
        variables.V_good_default[a_i, e_i, z_i] = a < -η*y ? utility_function(c_default, σ) + β * V_bad_expect[1, e_i] : -Inf
        # good credit history
        V_max = max(variables.V_good_repay[a_i, e_i, z_i], variables.V_good_default[a_i, e_i, z_i])
        if V_max == -Inf
            variables.V_good[a_i, e_i, z_i] = V_max
            variables.policy_good_d[a_i, e_i, z_i] = 1.0
        else
            V_good_sum = exp((variables.V_good_repay[a_i, e_i, z_i] - V_max) / ζ_d) + exp((variables.V_good_default[a_i, e_i, z_i] - V_max) / ζ_d)
            variables.V_good[a_i, e_i, z_i] = V_max + ζ_d * log(V_good_sum)
            variables.policy_good_d[a_i, e_i, z_i] = exp((variables.V_good_default[a_i, e_i, z_i] - V_max) / ζ_d) / V_good_sum
        end

        # bad credit history
        if a_i >= a_ind_zero
            a_pos_i = a_i - a_ind_zero + 1
            c_bad = y .+ a .- variables.q[a_ind_zero:end, e_i] .* a_grid_pos
            # variables.V_bad_all[:, a_pos_i, e_i, z_i] = (1.0 - β) * utility_function.(c_bad, σ) .+ β * (p_h * V_good_expect[a_ind_zero:end, e_i] .+ (1.0 - p_h) * V_bad_expect[:, e_i])
            variables.V_bad_all[:, a_pos_i, e_i, z_i] = utility_function.(c_bad, σ) .+ β * (p_h * V_good_expect[a_ind_zero:end, e_i] .+ (1.0 - p_h) * V_bad_expect[:, e_i])
            V_bad_max = maximum(variables.V_bad_all[:, a_pos_i, e_i, z_i])
            if V_bad_max == -Inf
                variables.V_bad[a_pos_i, e_i, z_i] = V_bad_max
                variables.policy_bad_a[:, a_pos_i, e_i, z_i] .= 0.0
            else
                V_bad_exp = exp.((variables.V_bad_all[:, a_pos_i, e_i, z_i] .- V_bad_max) ./ ζ_a)
                V_bad_sum = sum(V_bad_exp)
                variables.V_bad[a_pos_i, e_i, z_i] = V_bad_max + ζ_a * log(V_bad_sum)
                variables.policy_bad_a[:, a_pos_i, e_i, z_i] = V_bad_exp ./ V_bad_sum
            end
        end
    end
end

function loan_price_function!(variables::Mutable_Variables, parameters::NamedTuple)
    """
    compute loan price
    """

    @unpack a_grid_neg, a_size_neg, e_grid, e_size, e_Γ, z_grid, z_size, z_Γ, η, r_f, τ = parameters

    for e_i = 1:e_size, a_p_i = 1:(a_size_neg-1)
        repayment_expect = 0.0
        for z_p_i = 1:z_size, e_p_i = 1:e_size
            repayment_expect += variables.policy_good_d[a_p_i, e_p_i, z_p_i] * η * e_Γ[e_i, e_p_i] * z_Γ[z_p_i]
            repayment_expect += (1.0 - variables.policy_good_d[a_p_i, e_p_i, z_p_i]) * e_Γ[e_i, e_p_i] * z_Γ[z_p_i] * (-a_grid_neg[a_p_i])
        end
        variables.q[a_p_i, e_i] = repayment_expect / ((1 + r_f + τ + variables.aggregate_prices.ι_λ) * (-a_grid_neg[a_p_i]))
    end
end

function solve_function!(variables::Mutable_Variables, parameters::NamedTuple; tol::Real = tol_h, iter_max::Integer = iter_max)
    """
    solve model
    """
    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving model (one-loop): ")

    # initialize the next-period functions
    V_good_p = similar(variables.V_good)
    V_bad_p = similar(variables.V_bad)
    q_p = similar(variables.q)

    while crit > tol && iter < iter_max

        # copy previous unconditional value and loan pricing functions
        copyto!(V_good_p, variables.V_good)
        copyto!(V_bad_p, variables.V_bad)
        copyto!(q_p, variables.q)

        # update value functions and choice probability
        value_function!(V_good_p, V_bad_p, variables, parameters)

        # compute loan price
        loan_price_function!(variables, parameters)

        # check convergence
        V_good_diff = norm(variables.V_good .- V_good_p, Inf)
        V_bad_diff = norm(variables.V_bad .- V_bad_p, Inf)
        q_diff = norm(variables.q .- q_p, Inf)
        crit = max(V_good_diff, V_bad_diff, q_diff)

        # report progress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end

parameters = parameters_function()
variables = variables_function(parameters; λ = 0.0)
solve_function!(variables, parameters; tol = 1E-6, iter_max = 1000)
