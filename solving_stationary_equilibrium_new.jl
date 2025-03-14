#=============================#
# Solve stationary equlibrium #
#=============================#

function adda_cooper(N::Int64, ρ::Float64, σ::Float64; μ::Float64=0.0)
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
            integral = quadgk(u -> f(u), ϵ[i], ϵ[i+1])[1]
            Π[i, j] = (N / sqrt(2.0 * π * σ_ϵ^2.0)) * integral
        end
    end
    return z, Π
end

function parameters_function(;
    β::Float64=0.955,                # discount factor (households)
    ρ::Float64=0.975,                # survival rate
    r_f::Float64=0.04,               # risk-free rate # 1.04*ρ-1.0
    # r_f::Float64=1.04/ρ-1.0,         # risk-free rate # 1.04*ρ-1.0
    β_f::Float64=1.0 / (1.0 + r_f),  # discount factor (bank)
    # τ::Float64=0.00,                 # transaction cost
    τ::Float64=0.04,                 # transaction cost
    σ::Float64=2.00,                 # CRRA coefficient
    δ::Float64=0.08,                 # depreciation rate
    α::Float64=0.33,                 # capital share
    ψ::Float64=0.972^4,              # exogenous retention ratio # 1.0 - 1.0 / 20.0
    θ::Float64=1.0 / (4.57 * 0.75),  # diverting fraction # 1.0 / 3.0
    p_h::Float64=1.0 / 6.0,          # prob. of history erased
    η::Float64=0.40,                 # wage garnishment rate
    ξ::Float64=0.00,                 # stigma utility filing cost
    κ::Float64=697 / 33176,          # out-of-pocket monetary filing cost
    e_1_σ::Float64=0.448,            # s.d. of permanent endowment shock
    e_1_size::Int64=2,               # number of permanent endowment shock
    e_2_ρ::Float64=0.957,            # AR(1) of persistent endowment shock
    e_2_σ::Float64=0.129,            # s.d. of persistent endowment shock
    e_2_size::Int64=9,              # number of pesistent endowment shock
    e_3_σ::Float64=0.351,            # s.d. of transitory endowment shock
    e_3_size::Int64=3,               # number of transitory endowment shock
    ν_size::Int64=3,                 # number of expenditure shock
    a_min::Float64=-3.0,             # min of asset holding
    a_max::Float64=800.0,            # max of asset holding
    a_size_neg::Int64=501,           # number of grid of negative asset holding for VFI
    a_size_pos::Int64=101,           # number of grid of positive asset holding for VFI
    a_degree::Int64=3,               # curvature of positive asset gridpoints
    μ_scale::Int64=1                 # scale for the asset holding gridpoints for distribution
)
    """
    contruct an immutable object containg all paramters
    """

    # permanent endowment shock
    e_1_grid, e_1_Γ = adda_cooper(e_1_size, 0.0, e_1_σ)
    e_1_grid = exp.(e_1_grid)
    e_1_Γ = e_1_Γ[1, :]
    # e_1_grid = [-e_1_σ, e_1_σ]
    # e_1_Γ = Matrix(1.0I, e_1_size, e_1_size)
    # G_e_1 = [1.0 / e_1_size for i = 1:e_1_size]
    G_e_1 = e_1_Γ

    # persistent endowment shock
    # e_2_MC = tauchen(e_2_size, e_2_ρ, e_2_σ, 0.0, 3)
    # e_2_MC = rouwenhorst(e_2_size, e_2_ρ, e_2_σ, 0.0)
    # e_2_Γ = e_2_MC.p
    # e_2_grid = collect(e_2_MC.state_values)
    e_2_grid, e_2_Γ = adda_cooper(e_2_size, e_2_ρ, e_2_σ)
    e_2_grid = exp.(e_2_grid)
    G_e_2 = stationary_distributions(MarkovChain(e_2_Γ, e_2_grid))[1]
    # G_e_2 = [1.0, 0.0, 0.0]

    # transitory endowment shock
    e_3_grid, e_3_Γ = adda_cooper(e_3_size, 0.0, e_3_σ)
    e_3_grid = exp.(e_3_grid)
    e_3_Γ = e_3_Γ[1, :]
    # e_3_bar = sqrt((3 / 2) * e_3_σ^2)
    # e_3_grid = [-e_3_bar, 0.0, e_3_bar]
    # e_3_Γ = [1.0 / e_3_size for i = 1:e_3_size]
    G_e_3 = e_3_Γ # [0.0, 1.0, 0.0]

    # aggregate labor endowment
    E = 1.0

    # expenditure schock
    # ν_grid = zeros(ν_size)
    if ν_size == 3
        # ν_grid = [0.0, 0.3584239, 3.0]
        # ν_p_1 = 0.04438342
        # ν_p_2 = 0.0002092103
        # ν_Γ = [1.0 - ν_p_1 - ν_p_2, ν_p_1, ν_p_2]

        ν_grid = [0.0, 0.0, 0.0]
        ν_p_1 = 0.0
        ν_p_2 = 0.0
        ν_Γ = [1.0 - ν_p_1 - ν_p_2, ν_p_1, ν_p_2]
    elseif ν_size == 4
        ν_size = 3
        ν_grid = [0.0, 0.3584239, 3.0]
        ν_p_1 = 0.04438342
        ν_p_2 = 0.0
        ν_Γ = [1.0 - ν_p_1 - ν_p_2, ν_p_1, ν_p_2]
    elseif ν_size == 5
        ν_size = 3
        ν_grid = [0.0, 0.3584239, 3.0]
        ν_p_1 = 0.0
        ν_p_2 = 0.0
        ν_Γ = [1.0 - ν_p_1 - ν_p_2, ν_p_1, ν_p_2]
    elseif ν_size == 6
        ν_size = 3
        ν_grid = [0.0, 0.3584239, 3.0]
        ν_p_1 = 0.04438342 * 0.9
        ν_p_2 = 0.0
        ν_Γ = [1.0 - ν_p_1 - ν_p_2, ν_p_1, ν_p_2]
    else
        ν_grid = [0.0, 0.3584239]
        ν_p = 0.04705877
        ν_Γ = [1.0 - ν_p, ν_p]
    end
    G_ν = ν_Γ

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length=a_size_neg))
    a_grid_pos = ((range(0.0, stop=a_size_pos - 1, length=a_size_pos) / (a_size_pos - 1)) .^ a_degree) * a_max
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims=1)
    a_size = length(a_grid)
    a_ind_zero = a_size_neg

    # asset holding grid for μ
    a_size_neg_μ = a_size_neg * μ_scale
    a_size_pos_μ = a_size_pos * μ_scale
    a_grid_neg_μ = collect(range(a_min, 0.0, length=a_size_neg_μ))
    a_grid_pos_μ = collect(range(0.0, a_max, length=a_size_pos_μ))
    # a_grid_pos_μ = ((range(0.0, stop=a_size_pos_μ - 1, length=a_size_pos_μ) / (a_size_pos_μ - 1)) .^ a_degree) * a_max
    a_grid_μ = cat(a_grid_neg_μ[1:(end-1)], a_grid_pos_μ, dims=1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(iszero, a_grid_μ)[]

    # iterators
    # loop_V = collect(Iterators.product(1:ν_size, 1:e_3_size, 1:e_2_size, 1:e_1_size, 1:a_size))
    loop_V = collect(Iterators.product(1:ν_size, 1:e_3_size, 1:a_size))
    loop_EV = collect(Iterators.product(1:e_2_size, 1:e_1_size, 1:a_size))
    loop_EV_d = collect(Iterators.product(1:e_3_size, 1:e_2_size, 1:e_1_size))
    loop_q = collect(Iterators.product(1:e_2_size, 1:e_1_size, 1:(a_size_neg-1)))
    loop_q_p = collect(Iterators.product(1:ν_size, 1:e_3_size, 1:e_2_size))
    loop_rbl = collect(Iterators.product(1:e_2_size, 1:e_1_size))
    loop_μ = collect(Iterators.product(1:ν_size, 1:e_3_size, 1:e_2_size, 1:e_1_size))

    # return values
    return (
        β=β,
        ρ=ρ,
        r_f=r_f,
        β_f=β_f,
        τ=τ,
        σ=σ,
        δ=δ,
        α=α,
        ψ=ψ,
        θ=θ,
        p_h=p_h,
        η=η,
        ξ=ξ,
        κ=κ,
        e_1_σ=e_1_σ,
        e_1_size=e_1_size,
        e_1_Γ=e_1_Γ,
        e_1_grid=e_1_grid,
        G_e_1=G_e_1,
        e_2_ρ=e_2_ρ,
        e_2_σ=e_2_σ,
        e_2_size=e_2_size,
        e_2_Γ=e_2_Γ,
        e_2_grid=e_2_grid,
        G_e_2=G_e_2,
        e_3_σ=e_3_σ,
        e_3_size=e_3_size,
        e_3_Γ=e_3_Γ,
        e_3_grid=e_3_grid,
        G_e_3=G_e_3,
        E=E,
        ν_size=ν_size,
        ν_Γ=ν_Γ,
        ν_grid=ν_grid,
        G_ν=G_ν,
        a_grid=a_grid,
        a_grid_neg=a_grid_neg,
        a_grid_pos=a_grid_pos,
        a_size=a_size,
        a_size_neg=a_size_neg,
        a_size_pos=a_size_pos,
        a_ind_zero=a_ind_zero,
        a_grid_μ=a_grid_μ,
        a_grid_neg_μ=a_grid_neg_μ,
        a_grid_pos_μ=a_grid_pos_μ,
        a_size_μ=a_size_μ,
        a_size_neg_μ=a_size_neg_μ,
        a_size_pos_μ=a_size_pos_μ,
        a_ind_zero_μ=a_ind_zero_μ,
        a_degree=a_degree,
        loop_V=loop_V,
        loop_EV=loop_EV,
        loop_EV_d=loop_EV_d,
        loop_q=loop_q,
        loop_q_p=loop_q_p,
        loop_rbl=loop_rbl,
        loop_μ=loop_μ,
    )
end

mutable struct Mutable_Aggregate_Prices
    """
    construct a type for mutable aggregate prices
    """
    λ::Float64
    ξ_λ::Float64
    Λ_λ::Float64
    leverage_ratio_λ::Float64
    KL_to_D_ratio_λ::Float64
    ι_λ::Float64
    r_k_λ::Float64
    K_λ::Float64
    w_λ::Float64
end

mutable struct Mutable_Aggregate_Variables
    """
    construct a type for mutable aggregate variables
    """
    K::Float64
    L::Float64
    L_adj::Float64
    D::Float64
    N::Float64
    profit::Float64
    ω::Float64
    leverage_ratio::Float64
    KL_to_D_ratio::Float64
    debt_to_earning_ratio::Float64
    share_of_filers::Float64
    share_of_involuntary_filers::Float64
    share_in_debts::Float64
    avg_loan_rate::Float64
    avg_loan_rate_pw::Float64
end

mutable struct Mutable_Variables
    """
    construct a type for mutable variables
    """
    aggregate_prices::Mutable_Aggregate_Prices
    aggregate_variables::Mutable_Aggregate_Variables
    R::Array{Float64,3}
    q::Array{Float64,3}
    rbl::Array{Float64,3}
    V::Array{Float64,5}
    E_V::Array{Float64,3}
    V_d::Array{Float64,3}
    u_c_d::Array{Float64,3}
    V_nd::Array{Float64,5}
    V_pos::Array{Float64,5}
    E_V_pos::Array{Float64,3}
    policy_a::Array{Float64,5}
    policy_d::Array{Float64,5}
    policy_a_pos::Array{Float64,5}
    policy_d_pos::Array{Float64,5}
    μ::Array{Float64,6}
end

function min_bounds_function(obj::Function, grid_min::Float64, grid_max::Float64; grid_length::Int64=120, obj_range::Int64=1)
    """
    compute bounds for minimization
    """

    grid = range(grid_min, grid_max, length=grid_length)
    obj_grid = obj.(grid)
    obj_index = argmin(obj_grid)
    if obj_index < (1 + obj_range)
        lb = grid_min
        ub = grid[obj_index+obj_range]
    elseif obj_index > (grid_length - obj_range)
        lb = grid[obj_index-obj_range]
        ub = grid_max
    else
        lb = grid[obj_index-obj_range]
        ub = grid[obj_index+obj_range]
    end
    return lb, ub
end

function min_bounds_function!(bb::Vector{Float64}, grid::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64}, obj::Function, grid_min::Float64, grid_max::Float64; grid_length::Int64=120, obj_range::Int64=1)
    """
    compute bounds for minimization
    """

    grid .= range(grid_min, grid_max, length=grid_length)
    grid_size = length(grid)
    obj_grid = obj.(grid)
    obj_index = argmin(obj_grid)
    if obj_index < (1 + obj_range)
        bb[1] = grid_min
        bb[2] = grid[obj_index+obj_range]
    elseif obj_index > (grid_size - obj_range)
        bb[1] = grid[obj_index-obj_range]
        bb[2] = grid_max
    else
        bb[1] = grid[obj_index-obj_range]
        bb[2] = grid[obj_index+obj_range]
    end
    return nothing
end

function utility_function(c::Float64, γ::Float64)
    """
    compute utility of CRRA utility function with coefficient γ
    """

    if c > 0.0
        return γ == 1.0 ? log(c) : 1.0 / ((1.0 - γ) * c^(γ - 1.0))
    else
        return -Inf
    end
end

function repayment_function(e_1_i::Int64, e_2_i::Int64, e_3_p_i::Int64, a_p::Float64, threshold_e_2::Float64, w::Float64, parameters::NamedTuple; wage_garnishment::Bool=true)
    """
    evaluate repayment recovery rate with wage garnishment
    """

    # unpack parameters
    @unpack e_1_grid, e_2_grid, e_3_grid, e_2_ρ, e_2_σ, η = parameters

    # permanent and transitory components
    e_1 = e_1_grid[e_1_i]
    e_3 = e_3_grid[e_3_p_i]

    # compute expected repayment amount
    e_2_μ = e_2_ρ * e_2_grid[e_2_i]

    # (1) not default
    # default_prob = cdf(Normal(e_2_μ, e_2_σ), threshold_e_2)
    # amount_repay = -a_p * (1.0 - default_prob)

    # (2) default and reclaiming wage garnishment is enabled
    amount_default = 0.0
    if wage_garnishment == true
        default_adjusted_prob = cdf(Normal(e_2_μ + e_2_σ^2.0, e_2_σ), threshold_e_2)
        amount_default = η * w * exp(e_1 + e_3) * exp(e_2_μ + e_2_σ^2.0 / 2.0) * default_adjusted_prob
        amount_repay = -a_p * (1.0 - default_adjusted_prob)
    end

    # (3) total amount collected by banks
    total_amount = amount_repay + amount_default
    total_amount = clamp(total_amount, 0, -a_p)

    return total_amount
end

function aggregate_prices_λ_funtion(parameters::NamedTuple; λ::Float64)
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

function variables_function(parameters::NamedTuple; λ::Float64, load_init::Bool=false)
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack a_ind_zero, a_size, a_grid, a_size_pos, a_size_neg, a_grid_neg, a_ind_zero_μ, a_size_μ, a_size_pos_μ = parameters
    @unpack e_1_size, e_1_grid, e_1_Γ, e_2_size, e_2_grid, e_2_Γ, e_2_ρ, e_2_σ, e_3_size, e_3_grid, e_3_Γ = parameters
    @unpack ν_size, ν_Γ = parameters
    @unpack ρ, r_f, τ, η, κ, σ = parameters

    # define aggregate prices
    ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ = aggregate_prices_λ_funtion(parameters; λ=λ)
    aggregate_prices = Mutable_Aggregate_Prices(λ, ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ)

    # define aggregate variables
    K = K_λ
    L = 0.0
    L_adj = 0.0
    D = 0.0
    N = 0.0
    profit = 0.0
    ω = 0.0
    leverage_ratio = 0.0
    KL_to_D_ratio = 0.0
    debt_to_earning_ratio = 0.0
    share_of_filers = 0.0
    share_of_involuntary_filers = 0.0
    share_in_debts = 0.0
    avg_loan_rate = 0.0
    avg_loan_rate_pw = 0.0
    aggregate_variables = Mutable_Aggregate_Variables(K, L, L_adj, D, N, profit, ω, leverage_ratio, KL_to_D_ratio, debt_to_earning_ratio, share_of_filers, share_of_involuntary_filers, share_in_debts, avg_loan_rate, avg_loan_rate_pw)

    if load_init == false

        # define repayment probability, pricing function, and risky borrowing limit
        R = ones(a_size_neg, e_1_size, e_2_size)
        q = ones(a_size, e_1_size, e_2_size) .* ρ ./ (1.0 + r_f)
        rbl = zeros(e_1_size, e_2_size, 2)

        # define value and policy functions
        V = zeros(a_size, e_1_size, e_2_size, e_3_size, ν_size)
        E_V = zeros(a_size, e_1_size, e_2_size)
        V_d = zeros(e_1_size, e_2_size, e_3_size)
        u_c_d = zeros(e_1_size, e_2_size, e_3_size)
        for e_3_i = 1:e_3_size, e_2_i = 1:e_2_size, e_1_i = 1:e_1_size
            u_c_d[e_1_i, e_2_i, e_3_i] = utility_function((1.0 - η) * w_λ * e_1_grid[e_1_i] * e_2_grid[e_2_i] * e_3_grid[e_3_i] - κ, σ)
        end
        V_nd = zeros(a_size, e_1_size, e_2_size, e_3_size, ν_size)
        V_pos = zeros(a_size_pos, e_1_size, e_2_size, e_3_size, ν_size)
        E_V_pos = zeros(a_size_pos, e_1_size, e_2_size)

        # define cross-sectional distribution
        μ = zeros(a_size_μ, e_1_size, e_2_size, e_3_size, ν_size, 2)
        μ_size = (a_size_μ + a_size_pos_μ) * e_1_size * e_2_size * e_3_size * ν_size
        μ[:, :, :, :, :, 1] .= 1.0 ./ μ_size
        μ[a_ind_zero_μ:end, :, :, :, :, 2] .= 1.0 ./ μ_size
    else
        @load "results_int.jld2" V V_d V_nd V_pos R q rbl μ
    end
    policy_a = zeros(a_size, e_1_size, e_2_size, e_3_size, ν_size)
    policy_d = zeros(a_size, e_1_size, e_2_size, e_3_size, ν_size)
    policy_a_pos = zeros(a_size_pos, e_1_size, e_2_size, e_3_size, ν_size)
    policy_d_pos = zeros(a_size_pos, e_1_size, e_2_size, e_3_size, ν_size)

    # return outputs
    variables = Mutable_Variables(aggregate_prices, aggregate_variables, R, q, rbl, V, E_V, V_d, u_c_d, V_nd, V_pos, E_V_pos, policy_a, policy_d, policy_a_pos, policy_d_pos, μ)
    return variables
end

function variables_function_update!(variables::Mutable_Variables, parameters::NamedTuple; λ::Float64)
    """
    construct a mutable object containing endogenous variables
    """

    # define aggregate prices
    ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ = aggregate_prices_λ_funtion(parameters; λ=λ)
    variables.aggregate_prices = Mutable_Aggregate_Prices(λ, ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ)
end

function E_V_function!(E_V::Array{Float64,3}, E_V_pos::Array{Float64,3}, V_p::Array{Float64,5}, V_pos_p::Array{Float64,5}, parameters::NamedTuple)
    """
    construct expected value functions
    """

    # unpack parameters
    @unpack e_1_size, e_2_size, e_2_Γ, e_3_size, e_3_Γ, ν_size, ν_Γ, a_size, a_size_pos, a_ind_zero, ρ, β, loop_EV = parameters

    # update expected value 
    @batch for (e_2_i, e_1_i, a_p_i) in loop_EV
        if a_p_i < a_ind_zero
            E_V[a_p_i, e_1_i, e_2_i] = 0.0
            for ν_p_i = 1:ν_size, e_3_p_i = 1:e_3_size, e_2_p_i = 1:e_2_size
                E_V[a_p_i, e_1_i, e_2_i] += ρ * β * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * V_p[a_p_i, e_1_i, e_2_p_i, e_3_p_i, ν_p_i]
            end
        else
            E_V[a_p_i, e_1_i, e_2_i] = 0.0
            a_p_i_pos = a_p_i - a_ind_zero + 1
            E_V_pos[a_p_i_pos, e_1_i, e_2_i] = 0.0
            for ν_p_i = 1:ν_size, e_3_p_i = 1:e_3_size, e_2_p_i = 1:e_2_size
                E_V[a_p_i, e_1_i, e_2_i] += ρ * β * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * V_p[a_p_i, e_1_i, e_2_p_i, e_3_p_i, ν_p_i]
                E_V_pos[a_p_i_pos, e_1_i, e_2_i] += ρ * β * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * V_pos_p[a_p_i_pos, e_1_i, e_2_p_i, e_3_p_i, ν_p_i]
            end
        end
    end

    # replace NaN with -Inf
    # replace!(E_V, NaN => -Inf)
    # replace!(E_V_pos, NaN => -Inf)

    # return results
    return nothing
end

function V_d_function!(V_d::Array{Float64,3}, u_c_d::Array{Float64,3}, E_V_pos::Array{Float64,3}, parameters::NamedTuple)
    """
    update default value function
    """

    # unpack parameters
    @unpack e_1_size, e_2_size, e_3_size, ξ, loop_EV_d = parameters

    # update default value 
    for (e_3_i, e_2_i, e_1_i) in loop_EV_d
        V_d[e_1_i, e_2_i, e_3_i] = u_c_d[e_1_i, e_2_i, e_3_i] - ξ + E_V_pos[1, e_1_i, e_2_i]
    end

    # return results
    return nothing
end

function value_and_policy_function!(
    variables::Mutable_Variables,
    V_p::Array{Float64,5},
    V_pos_p::Array{Float64,5},
    parameters::NamedTuple
)
    """
    one-step update of value and policy functions
    """

    # unpack parameters
    @unpack a_size, a_grid, a_size_pos, a_grid_pos, a_ind_zero = parameters
    @unpack e_1_size, e_1_grid, e_1_Γ, e_2_size, e_2_grid, e_2_Γ, e_3_size, e_3_grid, e_3_Γ = parameters
    @unpack ν_size, ν_grid, ν_Γ = parameters
    @unpack ρ, β, σ, r_f = parameters
    @unpack p_h, η, κ, ξ = parameters
    @unpack loop_V = parameters

    # pre-compute the next-period discounted expected value funtions and defaulting value
    E_V_function!(variables.E_V, variables.E_V_pos, V_p, V_pos_p, parameters)
    V_d_function!(variables.V_d, variables.u_c_d, variables.E_V_pos, parameters)

    # create interpolation containers
    qa_function_itp = linear_interpolation(a_grid, a_grid, extrapolation_bc=Line())
    V_hat_itp = linear_interpolation(a_grid, a_grid, extrapolation_bc=Line())
    V_hat_pos_itp = linear_interpolation(a_grid_pos, a_grid_pos, extrapolation_bc=Line())

    # loop over all states
    # Threads.@threads for (ν_i, e_3_i, e_2_i, e_1_i, a_i) in loop_V
    for e_2_i = 1:e_2_size, e_1_i = 1:e_1_size

        # total permanent and persistent earnings
        e_12 = e_1_grid[e_1_i] * e_2_grid[e_2_i]

        # extract risky borrowing limit and maximum discounted borrowing amount
        @views rbl_a, rbl_qa = variables.rbl[e_1_i, e_2_i, :]

        # construct interpolated functions
        @views qa = variables.q[:, e_1_i, e_2_i] .* a_grid
        # qa_function_itp = Akima(a_grid, qa)
        # qa_function_itp = linear_interpolation(a_grid, qa, extrapolation_bc=Line())
        @views qa_function_itp.itp.coefs[:] = qa

        @views V_hat = variables.E_V[:, e_1_i, e_2_i]
        @views V_hat_pos = variables.E_V_pos[:, e_1_i, e_2_i]
        # V_hat_itp = Akima(a_grid, V_hat)
        # V_hat_itp = linear_interpolation(a_grid, V_hat, extrapolation_bc=Line())
        @views V_hat_itp.itp.coefs[:] = V_hat

        @views V_hat_pos_ = p_h * V_hat[a_ind_zero:end] + (1.0 - p_h) * V_hat_pos
        # V_hat_pos_itp = Akima(a_grid_pos, V_hat_pos_)
        # V_hat_pos_itp = linear_interpolation(a_grid_pos, V_hat_pos_, extrapolation_bc=Line())
        @views V_hat_pos_itp.itp.coefs[:] = V_hat_pos_

        # define objective functions
        object_nd(a_p, CoH) = -(utility_function(CoH - qa_function_itp(a_p), σ) + V_hat_itp(a_p))
        object_pos(a_p, CoH) = -(utility_function(CoH - qa_function_itp(a_p), σ) + V_hat_pos_itp(a_p))

        # for ν_i = 1:ν_size, e_3_i = 1:e_3_size, a_i = 1:a_size
        for (ν_i, e_3_i, a_i) in loop_V

            # constrcut cash on hand
            @views CoH = variables.aggregate_prices.w_λ * e_12 * e_3_grid[e_3_i] + a_grid[a_i] - ν_grid[ν_i]

            # good credit history
            if (CoH - rbl_qa) > 0.0
                object_nd_(a_p) = object_nd(a_p, CoH)
                res_nd = optimize(a_p -> object_nd_(a_p), rbl_a, CoH, GoldenSection())
                variables.V_nd[a_i, e_1_i, e_2_i, e_3_i, ν_i] = -Optim.minimum(res_nd)
                if variables.V_nd[a_i, e_1_i, e_2_i, e_3_i, ν_i] >= variables.V_d[e_1_i, e_2_i, e_3_i]
                    variables.V[a_i, e_1_i, e_2_i, e_3_i, ν_i] = variables.V_nd[a_i, e_1_i, e_2_i, e_3_i, ν_i]
                    variables.policy_a[a_i, e_1_i, e_2_i, e_3_i, ν_i] = Optim.minimizer(res_nd)
                    variables.policy_d[a_i, e_1_i, e_2_i, e_3_i, ν_i] = 0.0
                else
                    variables.V[a_i, e_1_i, e_2_i, e_3_i, ν_i] = variables.V_d[e_1_i, e_2_i, e_3_i]
                    variables.policy_a[a_i, e_1_i, e_2_i, e_3_i, ν_i] = 0.0
                    variables.policy_d[a_i, e_1_i, e_2_i, e_3_i, ν_i] = 1.0
                end
            else
                variables.V_nd[a_i, e_1_i, e_2_i, e_3_i, ν_i] = -Inf
                variables.V[a_i, e_1_i, e_2_i, e_3_i, ν_i] = variables.V_d[e_1_i, e_2_i, e_3_i]
                variables.policy_a[a_i, e_1_i, e_2_i, e_3_i, ν_i] = 0.0
                variables.policy_d[a_i, e_1_i, e_2_i, e_3_i, ν_i] = 1.0
            end

            # bad credit history
            if a_i >= a_ind_zero
                a_pos_i = a_i - a_ind_zero + 1
                if CoH > 0.0
                    object_pos_(a_p) = object_pos(a_p, CoH)
                    res_pos = optimize(a_p -> object_pos(a_p, CoH), 0.0, CoH, GoldenSection())
                    variables.V_pos[a_pos_i, e_1_i, e_2_i, e_3_i, ν_i] = -Optim.minimum(res_pos)
                    variables.policy_a_pos[a_pos_i, e_1_i, e_2_i, e_3_i, ν_i] = Optim.minimizer(res_pos)
                    variables.policy_d_pos[a_pos_i, e_1_i, e_2_i, e_3_i, ν_i] = 0.0
                else
                    variables.V_pos[a_pos_i, e_1_i, e_2_i, e_3_i, ν_i] = variables.V_d[e_1_i, e_2_i, e_3_i]
                    variables.policy_a_pos[a_pos_i, e_1_i, e_2_i, e_3_i, ν_i] = 0.0
                    variables.policy_d_pos[a_pos_i, e_1_i, e_2_i, e_3_i, ν_i] = 1.0
                end
            end
        end
    end

    # return results
    return nothing
end

function pricing_and_rbl_function!(R::Array{Float64,3}, q::Array{Float64,3}, rbl::Array{Float64,3}, w::Float64, ι::Float64, parameters::NamedTuple)
    """
    update pricing function and borrowing risky limit
    """

    # unpack parameters
    @unpack ρ, r_f, τ, η = parameters
    @unpack a_ind_zero, a_size, a_grid, a_size_neg, a_grid_neg = parameters
    @unpack e_1_size, e_1_grid, e_1_Γ, e_2_size, e_2_grid, e_2_Γ, e_3_size, e_3_grid, e_3_Γ = parameters
    @unpack ν_size, ν_Γ, ν_grid = parameters
    @unpack loop_q, loop_q_p, loop_rbl = parameters

    # loop over states
    @batch for (e_2_i, e_1_i, a_p_i) in loop_q
        R[a_p_i, e_1_i, e_2_i] = 0.0
        # q[a_p_i, e_1_i, e_2_i] = 0.0
        a_p = a_grid_neg[a_p_i]
        for (ν_p_i, e_3_p_i, e_2_p_i) in loop_q_p
            e_p = e_1_grid[e_1_i] * e_2_grid[e_2_p_i] * e_3_grid[e_3_p_i]
            ν_p = ν_grid[ν_p_i]
            R[a_p_i, e_1_i, e_2_i] += e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * (1.0 - variables.policy_d[a_p_i, e_1_i, e_2_p_i, e_3_p_i, ν_p_i] + variables.policy_d[a_p_i, e_1_i, e_2_p_i, e_3_p_i, ν_p_i] * η * w * e_p / (ν_p - a_p))
        end
    end
    clamp!(R, 0.0, 1.0)
    q[1:a_size_neg, :, :] .= ρ .* R ./ (1.0 + r_f + τ + ι)

    for (e_2_i, e_1_i) in loop_rbl
        # risky borrowing limit and maximum discounted borrwoing amount
        # qa_function_itp = Akima(a_grid_neg, q[1:a_ind_zero, e_1_i, e_2_i] .* a_grid_neg)
        # qa_function_itp = Spline1D(a_grid_neg, q[1:a_ind_zero, e_1_i, e_2_i] .* a_grid_neg; k = 1, bc = "extrapolate")
        # qa_function(a_p) = qa_function_itp(a_p)
        # rbl_lb, rbl_ub = min_bounds_function(qa_function, a_grid_neg[1], 0.0)
        # res_rbl = optimize(qa_function, rbl_lb, rbl_ub)
        # res_rbl = optimize(qa_function, a_grid_neg[1], 0.0, GoldenSection())
        # rbl[e_1_i, e_2_i, 1] = Optim.minimizer(res_rbl)
        # rbl[e_1_i, e_2_i, 2] = Optim.minimum(res_rbl)
        res_rbl = findmin(q[1:a_ind_zero, e_1_i, e_2_i] .* a_grid_neg)
        rbl[e_1_i, e_2_i, 1] = a_grid_neg[res_rbl[2]]
        rbl[e_1_i, e_2_i, 2] = res_rbl[1]
    end

    # return results
    return nothing
end

function solve_value_and_pricing_function!(variables::Mutable_Variables, parameters::NamedTuple; tol::Float64=1E-8, iter_max::Int64=1000, slow_updating::Float64=1.0)
    """
    solve household and banking problems using one-loop algorithm
    """

    # initialize the iteration number and criterion
    search_iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving household and banking problems (one-loop): ")

    # construct containers
    V_p = similar(variables.V)
    V_pos_p = similar(variables.V_pos)
    q_p = similar(variables.q)

    while crit > tol && search_iter < iter_max

        # copy previous values
        copyto!(V_p, variables.V)
        copyto!(V_pos_p, variables.V_pos)
        copyto!(q_p, variables.q)

        # value and policy functions
        value_and_policy_function!(variables, V_p, V_pos_p, parameters)

        # pricing function and borrowing risky limit
        pricing_and_rbl_function!(variables.R, variables.q, variables.rbl, variables.aggregate_prices.w_λ, variables.aggregate_prices.ι_λ, parameters)

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

    return crit
end

function stationary_distribution_function!(variables::Mutable_Variables, μ_p::Array{Float64,6}, parameters::NamedTuple)
    """
    update stationary distribution
    """

    # unpack parameters
    @unpack e_1_size, e_1_Γ, G_e_1, e_2_size, e_2_Γ, G_e_2, e_3_size, e_3_Γ, G_e_3, ν_size, ν_Γ, G_ν, a_grid, a_grid_pos, a_size_μ, a_grid_μ, a_ind_zero_μ, ρ, p_h, loop_μ = parameters

    # initialize μ
    variables.μ .= 0.0

    # looping over current states
    for (ν_i, e_3_i, e_2_i, e_1_i) in loop_μ
        # for ν_i = 1:ν_size, e_3_i = 1:e_3_size, e_2_i = 1:e_2_size, e_1_i = 1:e_1_size  

        # interpolated decision rules
        policy_a_itp = Akima(a_grid, variables.policy_a[:, e_1_i, e_2_i, e_3_i, ν_i])
        policy_d_itp = Akima(a_grid, variables.policy_d[:, e_1_i, e_2_i, e_3_i, ν_i])
        policy_a_pos_itp = Akima(a_grid_pos, variables.policy_a_pos[:, e_1_i, e_2_i, e_3_i, ν_i])
        policy_d_pos_itp = Akima(a_grid_pos, variables.policy_d_pos[:, e_1_i, e_2_i, e_3_i, ν_i])

        # loop over the dimension of asset holding
        for a_μ_i = 1:a_size_μ

            # extract wealth and compute asset choice
            a_μ = a_grid_μ[a_μ_i]
            a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])
            d_a_μ = policy_d_itp(a_μ)

            # locate it on the original grid
            a_p_lb = findall(a_grid_μ .<= a_p)[end]
            a_p_ub = findall(a_p .<= a_grid_μ)[1]

            # compute weights
            if a_p_lb != a_p_ub
                a_p_lower = a_grid_μ[a_p_lb]
                a_p_upper = a_grid_μ[a_p_ub]
                weight_lower = (a_p_upper - a_p) / (a_p_upper - a_p_lower)
                weight_upper = (a_p - a_p_lower) / (a_p_upper - a_p_lower)
            else
                weight_lower = 0.5
                weight_upper = 0.5
            end

            # loop over the dimension of exogenous future individual states
            for (ν_p_i, e_3_p_i, e_2_p_i, e_1_p_i) in loop_μ
                # for ν_p_i = 1:ν_size, e_3_p_i = 1:e_3_size, e_2_p_i = 1:e_2_size, e_1_p_i = 1:e_1_size 
                if e_1_p_i == e_1_i
                    variables.μ[a_p_lb, e_1_i, e_2_p_i, e_3_p_i, ν_p_i, 1] += (1.0 - d_a_μ) * ρ * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * weight_lower * μ_p[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1]
                    variables.μ[a_p_ub, e_1_i, e_2_p_i, e_3_p_i, ν_p_i, 1] += (1.0 - d_a_μ) * ρ * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * weight_upper * μ_p[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1]
                    variables.μ[a_ind_zero_μ, e_1_i, e_2_p_i, e_3_p_i, ν_p_i, 2] += d_a_μ * ρ * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * μ_p[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1]
                end
                variables.μ[a_ind_zero_μ, e_1_p_i, e_2_p_i, e_3_p_i, ν_p_i, 1] += (1.0 - ρ) * G_e_1[e_1_p_i] * G_e_2[e_2_p_i] * G_e_3[e_3_p_i] * G_ν[ν_p_i] * μ_p[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1]
            end

            # for those with a bad history
            if a_μ >= 0.0
                a_p = clamp(policy_a_pos_itp(a_μ), 0.0, a_grid[end])
                d_a_μ = policy_d_pos_itp(a_μ)

                a_p_lb = findall(a_grid_μ .<= a_p)[end]
                a_p_ub = findall(a_p .<= a_grid_μ)[1]
                if a_p_lb != a_p_ub
                    a_p_lower = a_grid_μ[a_p_lb]
                    a_p_upper = a_grid_μ[a_p_ub]
                    weight_lower = (a_p_upper - a_p) / (a_p_upper - a_p_lower)
                    weight_upper = (a_p - a_p_lower) / (a_p_upper - a_p_lower)
                else
                    weight_lower = 0.5
                    weight_upper = 0.5
                end

                for (ν_p_i, e_3_p_i, e_2_p_i, e_1_p_i) in loop_μ
                    # for ν_p_i = 1:ν_size, e_3_p_i = 1:e_3_size, e_2_p_i = 1:e_2_size, e_1_p_i = 1:e_1_size
                    if e_1_p_i == e_1_i
                        variables.μ[a_p_lb, e_1_i, e_2_p_i, e_3_p_i, ν_p_i, 1] += (1.0 - d_a_μ) * ρ * p_h * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * weight_lower * μ_p[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 2]
                        variables.μ[a_p_ub, e_1_i, e_2_p_i, e_3_p_i, ν_p_i, 1] += (1.0 - d_a_μ) * ρ * p_h * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * weight_upper * μ_p[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 2]
                        variables.μ[a_p_lb, e_1_i, e_2_p_i, e_3_p_i, ν_p_i, 2] += (1.0 - d_a_μ) * ρ * (1.0 - p_h) * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * weight_lower * μ_p[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 2]
                        variables.μ[a_p_ub, e_1_i, e_2_p_i, e_3_p_i, ν_p_i, 2] += (1.0 - d_a_μ) * ρ * (1.0 - p_h) * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * weight_upper * μ_p[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 2]
                        variables.μ[a_ind_zero_μ, e_1_i, e_2_p_i, e_3_p_i, ν_p_i, 2] += d_a_μ * ρ * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * μ_p[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 2]
                    end
                    variables.μ[a_ind_zero_μ, e_1_p_i, e_2_p_i, e_3_p_i, ν_p_i, 1] += (1.0 - ρ) * G_e_1[e_1_p_i] * G_e_2[e_2_p_i] * G_e_3[e_3_p_i] * G_ν[ν_p_i] * μ_p[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 2]
                end
            end
        end
    end

    # standardize distribution
    sum_μ = sum(variables.μ)
    # println("sum_μ = $sum_μ")
    variables.μ .= variables.μ ./ sum_μ

    # return result
    return nothing
end

function solve_stationary_distribution_function!(variables::Mutable_Variables, parameters::NamedTuple; tol::Float64=1E-8, iter_max::Int64=2000)
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
        stationary_distribution_function!(variables, μ_p, parameters)

        # check convergence
        crit = norm(variables.μ .- μ_p, Inf)

        # update the iteration number
        search_iter += 1

        # manually report convergence progress
        # println("Solving stationary distribution: search_iter = $search_iter and crit = $crit > tol = $tol")
        ProgressMeter.update!(prog, crit)
    end

    return crit
end

function solve_aggregate_variable_function!(
    variables::Mutable_Variables,
    parameters::NamedTuple,
)
    """
    compute equlibrium aggregate variables
    """

    # unpack parameters
    @unpack e_1_size, e_1_grid, e_2_size, e_2_grid, e_3_size, e_3_grid, ν_size, a_grid, a_grid_neg, a_grid_pos, a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ, a_grid_μ, a_size_μ, r_f, τ, ψ, η, loop_μ = parameters

    # containers
    avg_loan_rate_num, avg_loan_rate_den = 0.0, 0.0
    avg_loan_rate_pw_num, avg_loan_rate_pw_den = 0.0, 0.0
    debt_to_earning_ratio_num, debt_to_earning_ratio_den = 0.0, 0.0

    # total loans, deposits, share of filers, nad debt-to-earning ratio
    for (ν_i, e_3_i, e_2_i, e_1_i) in loop_μ

        # interpolated decision rules
        @views policy_a_itp = Akima(a_grid, variables.policy_a[:, e_1_i, e_2_i, e_3_i, ν_i])
        @views policy_d_itp = Akima(a_grid, variables.policy_d[:, e_1_i, e_2_i, e_3_i, ν_i])
        @views policy_a_pos_itp = Akima(a_grid_pos, variables.policy_a_pos[:, e_1_i, e_2_i, e_3_i, ν_i])
        @views policy_d_pos_itp = Akima(a_grid_pos, variables.policy_d_pos[:, e_1_i, e_2_i, e_3_i, ν_i])

        # interpolated discounted borrowing amount
        @views q_function_itp = Akima(a_grid, variables.q[:, e_1_i, e_2_i])
        @views qa_function_itp = Akima(a_grid, variables.q[:, e_1_i, e_2_i] .* a_grid)

        # earnings
        we = variables.aggregate_prices.w_λ * e_1_grid[e_1_i] * e_2_grid[e_2_i] * e_3_grid[e_3_i]

        # loop over the dimension of asset holding
        for a_μ_i = 1:a_size_μ

            # extract wealth
            a_μ = a_grid_μ[a_μ_i]

            # good history
            a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])
            d_a_μ = policy_d_itp(a_μ)

            # share of filers
            variables.aggregate_variables.share_of_filers += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * d_a_μ

            # share of involuntary filers
            if we + a_μ - variables.rbl[e_1_i, e_2_i, 2] < 0.0
                variables.aggregate_variables.share_of_involuntary_filers += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * d_a_μ
            end

            # debt-to-earning ratio (num)
            debt_to_earning_ratio_num += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (-a_μ)

            # loans repaid
            variables.aggregate_variables.L_adj += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * ((-a_μ) * (1.0 - d_a_μ) + d_a_μ * η * we)

            if a_p < 0.0
                q_a_p = q_function_itp(a_p)
                qa_a_p = qa_function_itp(a_p)

                # total loans
                variables.aggregate_variables.L += -variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (1.0 - d_a_μ) * qa_a_p

                # average loan rate
                avg_loan_rate_num += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (1.0 - d_a_μ) * (1.0 / q_a_p - 1.0)
                avg_loan_rate_den += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (1.0 - d_a_μ)

                # average loan rate (persons-weighted)
                avg_loan_rate_pw_num += (1.0 - d_a_μ) * (1.0 / q_a_p - 1.0)
                avg_loan_rate_pw_den += 1

            elseif a_p > 0.0
                qa_a_p = qa_function_itp(a_p)

                # total deposits
                variables.aggregate_variables.D += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (1.0 - d_a_μ) * qa_a_p

            end

            # bad history
            if a_μ >= 0.0
                a_pos_p = clamp(policy_a_pos_itp(a_μ), 0.0, a_grid[end])
                d_pos_a_μ = policy_d_pos_itp(a_μ)

                # share of filers
                variables.aggregate_variables.share_of_filers += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 2] * d_pos_a_μ

                if a_pos_p > 0.0
                    qa_pos_a_p = qa_function_itp(a_pos_p)
                    variables.aggregate_variables.D += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 2] * (1.0 - d_pos_a_μ) * qa_pos_a_p
                end
            end

            # debt-to-earning ratio (den)
            debt_to_earning_ratio_den += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * we
            debt_to_earning_ratio_den += variables.μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 2] * we
        end
    end

    # net worth
    variables.aggregate_variables.N = (variables.aggregate_variables.K + variables.aggregate_variables.L) - variables.aggregate_variables.D

    # exogenous dividend policy
    # profit = (1.0 + r_f + ι) * K + (1.0 + τ + ι) * L - (1.0 + r_f) * D
    variables.aggregate_variables.profit = variables.aggregate_prices.ι_λ * (variables.aggregate_variables.K + variables.aggregate_variables.L) + (1.0 + r_f) * variables.aggregate_variables.N
    # ω = (N - ψ * profit) / ((1.0 - ψ) * profit)
    # ω = N / (ψ * profit)
    # ω = (N - ψ * profit) / ((1.0 - ψ) * (K + L))
    variables.aggregate_variables.ω = ( variables.aggregate_variables.N - ψ *  variables.aggregate_variables.profit) / ( variables.aggregate_variables.K +  variables.aggregate_variables.L)
    # ω = N - ψ * profit

    # leverage ratio
    variables.aggregate_variables.leverage_ratio = ( variables.aggregate_variables.K +  variables.aggregate_variables.L) /  variables.aggregate_variables.N

    # capital-loan-to-deposit ratio
    variables.aggregate_variables.KL_to_D_ratio = ( variables.aggregate_variables.K +  variables.aggregate_variables.L) /  variables.aggregate_variables.D

    # debt-to-earning ratio
    # debt_to_earning_ratio = debt_to_earning_ratio_num / debt_to_earning_ratio_den
    # debt_to_earning_ratio = L / w
    variables.aggregate_variables.debt_to_earning_ratio = debt_to_earning_ratio_num / debt_to_earning_ratio_den

    # average loan rate
    variables.aggregate_variables.avg_loan_rate = avg_loan_rate_num / avg_loan_rate_den
    variables.aggregate_variables.avg_loan_rate_pw = avg_loan_rate_pw_num / avg_loan_rate_pw_den

    # share in debt
    variables.aggregate_variables.share_in_debts = sum(variables.μ[1:(a_ind_zero_μ-1), :, :, :, :, 1])

    # return results
    return nothing
end

function solve_aggregate_variable_across_HH_function(
    policy_a::Array{Float64,5},
    policy_d::Array{Float64,5},
    policy_pos_a::Array{Float64,5},
    policy_pos_d::Array{Float64,5},
    q::Array{Float64,3},
    μ::Array{Float64,6},
    w::Float64,
    parameters::NamedTuple,
)
    """
    compute equlibrium aggregate variables
    """

    # unpack parameters
    @unpack e_1_size, e_1_grid, e_2_size, e_2_grid, e_3_size, e_3_grid, ν_size, a_grid, a_grid_neg, a_grid_pos, a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ, a_grid_μ, a_size_μ, r_f, τ, ψ, η = parameters

    # initialize container
    debt_to_earning_ratio = 0.0
    debt_to_earning_ratio_permanent_low = 0.0
    debt_to_earning_ratio_permanent_high = 0.0

    debt_to_earning_ratio_num = 0.0
    debt_to_earning_ratio_num_permanent_low = 0.0
    debt_to_earning_ratio_num_permanent_high = 0.0

    share_of_filers = 0.0
    share_of_filers_permanent_low = 0.0
    share_of_filers_permanent_high = 0.0

    share_in_debts = 0.0
    share_in_debts_permanent_low = 0.0
    share_in_debts_permanent_high = 0.0

    avg_loan_rate = 0.0
    avg_loan_rate_num = 0.0
    avg_loan_rate_den = 0.0

    avg_loan_rate_permanent_low = 0.0
    avg_loan_rate_num_permanent_low = 0.0
    avg_loan_rate_den_permanent_low = 0.0

    avg_loan_rate_permanent_high = 0.0
    avg_loan_rate_num_permanent_high = 0.0
    avg_loan_rate_den_permanent_high = 0.0

    # total loans, deposits, share of filers, nad debt-to-earning ratio
    for e_1_i = 1:e_1_size, e_2_i = 1:e_2_size, e_3_i = 1:e_3_size, ν_i = 1:ν_size

        # interpolated decision rules
        @views policy_a_Non_Inf = findall(policy_a[:, e_1_i, e_2_i, e_3_i, ν_i] .!= -Inf)
        policy_a_itp = Akima(a_grid[policy_a_Non_Inf], policy_a[policy_a_Non_Inf, e_1_i, e_2_i, e_3_i, ν_i])
        policy_d_itp = Akima(a_grid, policy_d[:, e_1_i, e_2_i, e_3_i, ν_i])
        policy_pos_a_itp = Akima(a_grid_pos, policy_pos_a[:, e_1_i, e_2_i, e_3_i, ν_i])
        policy_pos_d_itp = Akima(a_grid_pos, policy_pos_d[:, e_1_i, e_2_i, e_3_i, ν_i])

        # interpolated discounted borrowing amount
        @views q_e = q[:, e_1_i, e_2_i]
        q_function_itp = Akima(a_grid, q_e)

        # loop over the dimension of asset holding
        for a_μ_i = 1:a_size_μ

            # extract wealth and compute asset choice
            a_μ = a_grid_μ[a_μ_i]
            a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])

            if a_p < 0.0
                # average loan rate
                avg_loan_rate_num += μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ)) * (1.0 / q_function_itp(a_p) - 1.0)
                avg_loan_rate_den += μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ))
                if (e_1_i == 1) && (e_2_i == 2)
                    avg_loan_rate_num_permanent_low += μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ)) * (1.0 / q_function_itp(a_p) - 1.0)
                    avg_loan_rate_den_permanent_low += μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ))
                end
                if (e_1_i == 2) && (e_2_i == 2)
                    avg_loan_rate_num_permanent_high += μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ)) * (1.0 / q_function_itp(a_p) - 1.0)
                    avg_loan_rate_den_permanent_high += μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ))
                end
            end

            if a_μ < 0.0
                # share of filers
                share_of_filers += (μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * policy_d_itp(a_μ))
                if (e_1_i == 1) && (e_2_i == 2)
                    share_of_filers_permanent_low += (μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * policy_d_itp(a_μ)) / sum(μ[:, e_1_i, e_2_i, :, :, :])
                end
                if (e_1_i == 2) && (e_2_i == 2)
                    share_of_filers_permanent_high += (μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * policy_d_itp(a_μ)) / sum(μ[:, e_1_i, e_2_i, :, :, :])
                end

                # debt-to-earning ratio
                debt_to_earning_ratio_num += μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (-a_μ)
                if (e_1_i == 1) && (e_2_i == 2)
                    debt_to_earning_ratio_num_permanent_low += μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (-a_μ) / sum(μ[:, e_1_i, e_2_i, :, :, :])
                end
                if (e_1_i == 2) && (e_2_i == 2)
                    debt_to_earning_ratio_num_permanent_high += μ[a_μ_i, e_1_i, e_2_i, e_3_i, ν_i, 1] * (-a_μ) / sum(μ[:, e_1_i, e_2_i, :, :, :])
                end
            end
        end
    end

    # debt-to-earning ratio
    debt_to_earning_ratio = debt_to_earning_ratio_num / w
    debt_to_earning_ratio_permanent_low = debt_to_earning_ratio_num_permanent_low / (w * exp(e_1_grid[1]))
    debt_to_earning_ratio_permanent_high = debt_to_earning_ratio_num_permanent_high / (w * exp(e_1_grid[2]))

    # average loan rate
    avg_loan_rate = avg_loan_rate_num / avg_loan_rate_den
    avg_loan_rate_permanent_low = avg_loan_rate_num_permanent_low / avg_loan_rate_den_permanent_low
    avg_loan_rate_permanent_high = avg_loan_rate_num_permanent_high / avg_loan_rate_den_permanent_high

    # share in debt
    share_in_debts = sum(μ[1:(a_ind_zero_μ-1), :, :, :, :, 1])
    share_in_debts_permanent_low = sum(μ[1:(a_ind_zero_μ-1), 1, 2, :, :, 1]) ./ sum(μ[:, 1, 2, :, :, :])
    share_in_debts_permanent_high = sum(μ[1:(a_ind_zero_μ-1), 2, 2, :, :, 1]) ./ sum(μ[:, 2, 2, :, :, :])

    # return results
    return debt_to_earning_ratio, debt_to_earning_ratio_permanent_low, debt_to_earning_ratio_permanent_high, share_of_filers, share_of_filers_permanent_low, share_of_filers_permanent_high, share_in_debts, share_in_debts_permanent_low, share_in_debts_permanent_high, avg_loan_rate, avg_loan_rate_permanent_low, avg_loan_rate_permanent_high
end

function solve_economy_function!(variables::Mutable_Variables, parameters::NamedTuple; tol_h::Float64=1E-6, tol_μ::Float64=1E-8, slow_updating::Float64=1.0)
    """
    solve the economy with given liquidity multiplier ι
    """

    # solve household and banking problems
    crit_V = solve_value_and_pricing_function!(variables, parameters; tol=tol_h, iter_max=500, slow_updating=slow_updating)

    # solve the cross-sectional distribution
    crit_μ = solve_stationary_distribution_function!(variables, parameters; tol=tol_μ, iter_max=1000)

    # compute aggregate variables
    variables.aggregate_variables = solve_aggregate_variable_function(variables.policy_a, variables.threshold_a, variables.policy_pos_a, variables.policy_pos_d, variables.q, variables.rbl, variables.μ, variables.aggregate_prices.K_λ, variables.aggregate_prices.w_λ, variables.aggregate_prices.ι_λ, parameters)

    # compute the difference between demand and supply sides
    ED_KL_to_D_ratio = variables.aggregate_variables.KL_to_D_ratio - variables.aggregate_prices.KL_to_D_ratio_λ
    ED_leverage_ratio = variables.aggregate_variables.leverage_ratio - variables.aggregate_prices.leverage_ratio_λ

    # printout results
    data_spec = Any[
        "Effective Discount Factor" parameters.β variables.aggregate_variables.share_in_debts*100 40.14 #=1=#
        "Wage Garnishment Rate" parameters.η variables.aggregate_variables.share_of_filers*100 0.99 #=2=#
        "Bank Survival Rate" parameters.ψ variables.aggregate_variables.leverage_ratio 4.57 #=3=#
        "Diverting Fraction" parameters.θ variables.aggregate_variables.avg_loan_rate*100 9.26 #=4=#
        "Liquidity Multiplier" variables.aggregate_prices.λ "" "" #=5=#
        "Asset-to-Debt Ratio (Demand)" variables.aggregate_variables.KL_to_D_ratio "" "" #=6=#
        "Asset-to-Debt Ratio (Supply)" variables.aggregate_prices.KL_to_D_ratio_λ "" "" #=7=#
        "Difference" ED_KL_to_D_ratio "" "" #=8=#
        "Leverage Ratio (Demand)" variables.aggregate_variables.leverage_ratio "" "" #=9=#
        "Leverage Ratio (Supply)" variables.aggregate_prices.leverage_ratio_λ "" "" #=10=#
        "Difference" ED_leverage_ratio "" "" #=11=#
    ]
    pretty_table(data_spec; header=["Name", "Value", "Model Moment", "Data Moment"], alignment=[:l, :r, :r, :r], formatters=ft_round(8), body_hlines=[5, 8])

    # return excess demand
    return ED_KL_to_D_ratio, ED_leverage_ratio, crit_V, crit_μ
end

function optimal_multiplier_function(parameters::NamedTuple; λ_min_adhoc::Float64=-Inf, λ_max_adhoc::Float64=Inf, tol::Float64=1E-5, iter_max::Float64=200, slow_updating::Float64=1.0)
    """
    solve for optimal liquidity multiplier
    """

    # check the case of λ_min = 0.0
    λ_min = 0.0
    variables_λ_min = variables_function(parameters; λ=λ_min)
    ED_KL_to_D_ratio_λ_min, ED_leverage_ratio_λ_min, crit_V_min, crit_μ_min = solve_economy_function!(variables_λ_min, parameters; slow_updating=slow_updating)
    # if ED_KL_to_D_ratio_λ_min > 0.0
    #     return variables_λ_min, variables_λ_min, 1
    # end
    if ED_leverage_ratio_λ_min < 0.0
        return variables_λ_min, variables_λ_min, 1, crit_V_min, crit_μ_min
    end

    # check the case of λ_max = 1-ψ^(1/2)
    λ_max = 1.0 - sqrt(parameters.ψ)
    variables_λ_max = variables_function(parameters; λ=λ_max)
    ED_KL_to_D_ratio_λ_max, ED_leverage_ratio_λ_max, crit_V_max, crit_μ_max = solve_economy_function!(variables_λ_max, parameters; slow_updating=slow_updating)
    # if ED_KL_to_D_ratio_λ_max < 0.0
    #     return variables_λ_min, variables_λ_max, 2 # meaning solution doesn't exist!
    # end
    if ED_leverage_ratio_λ_max > 0.0
        return variables_λ_min, variables_λ_max, 2, crit_V_max, crit_μ_max # meaning solution doesn't exist!
    end

    # initialization
    search_iter = 0
    crit = Inf
    λ_optimal = 0.0
    crit_V_optimal = 0.0
    crit_μ_optimal = 0.0
    variables_λ_optimal = []
    λ_lower = max(λ_min_adhoc, λ_min)
    λ_upper = min(λ_max_adhoc, λ_max)

    # solve equlibrium multiplier by bisection
    while crit > tol && search_iter < iter_max

        # update the multiplier
        λ_optimal = (λ_lower + λ_upper) / 2

        # compute the associated results
        # if search_iter == 0
        #     variables_λ_optimal = variables_function(parameters; λ = λ_optimal)
        # else
        #     variables_function_update!(variables_λ_optimal, parameters; λ = λ_optimal)
        # end
        variables_λ_optimal = variables_function(parameters; λ=λ_optimal)
        ED_KL_to_D_ratio_λ_optimal, ED_leverage_ratio_λ_optimal, crit_V_optimal, crit_μ_optimal = solve_economy_function!(variables_λ_optimal, parameters; slow_updating=slow_updating)

        # update search region
        # if ED_KL_to_D_ratio_λ_optimal > 0.0
        #     λ_upper = λ_optimal
        # else
        #     λ_lower = λ_optimal
        # end
        if ED_leverage_ratio_λ_optimal < 0.0
            λ_upper = λ_optimal
        else
            λ_lower = λ_optimal
        end

        # check convergence
        # crit = abs(ED_KL_to_D_ratio_λ_optimal)
        crit = abs(ED_leverage_ratio_λ_optimal)

        # update the iteration number
        search_iter += 1

    end

    # return results
    return variables_λ_min, variables_λ_optimal, 3, crit_V_optimal, crit_μ_optimal
end
