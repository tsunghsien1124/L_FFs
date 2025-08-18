#=============================#
# Solve stationary equlibrium #
#=============================#

function adda_cooper(N::Int64, ρ::Float64, σ::Float64; μ::Float64=0.0)
    """
    Approximation of an autoregression process with a Markov chain proposed by Adda and Cooper (2003)
    """

    σ_ϵ = σ / sqrt(1.0 - ρ^2.0)
    ϵ = σ_ϵ .* quantile.(Normal(), range(0.0, 1.0, length=N + 1)) .+ μ
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

function initialize_parameters(;
    β::Float64=0.955,                # discount factor (households)
    ρ::Float64=0.975,                # survival rate
    r_f::Float64=0.04,               # risk-free rate # 1.04*ρ-1.0
    # r_f::Float64=1.04/ρ-1.0,         # risk-free rate # 1.04*ρ-1.0
    β_f::Float64=1.0 / (1.0 + r_f),  # discount factor (bank)
    # τ::Float64=0.00,                 # transaction cost
    τ::Float64=0.04,                 # transaction cost
    γ::Float64=2.00,                 # CRRA coefficient
    δ::Float64=0.08,                 # depreciation rate
    α::Float64=0.33,                 # capital share
    ψ::Float64=0.972^4,              # exogenous retention ratio # 1.0 - 1.0 / 20.0
    θ::Float64=1.0 / (4.57 * 0.75),  # diverting fraction # 1.0 / 3.0
    Ph::Float64=1.0 / 6.0,           # prob. of history erased
    η::Float64=0.40,                 # wage garnishment rate
    ξ::Float64=0.00,                 # stigma utility filing cost
    κ::Float64=697 / 33176,          # out-of-pocket monetary filing cost
    e1_σ::Float64=0.448,            # s.d. of permanent endowment shock
    e1_size::Int64=2,               # number of permanent endowment shock
    e2_ρ::Float64=0.957,            # AR(1) of persistent endowment shock
    e2_σ::Float64=0.129,            # s.d. of persistent endowment shock
    e2_size::Int64=5,               # number of persistent endowment shock
    e3_σ::Float64=0.351,            # s.d. of transitory endowment shock
    e3_size::Int64=3,               # number of transitory endowment shock
    ν_size::Int64=2,                 # number of preference shock
    # a_min::Float64=-5.0,             # min of asset holding
    a_max::Float64=800.0,            # max of asset holding
    a_size_neg::Int64=101,           # number of grid of negative asset holding for VFI
    a_size_pos::Int64=101,           # number of grid of positive asset holding for VFI
    a_degree::Int64=3,               # curvature of the positive asset gridpoints
    μ_scale::Int64=1,                # scale for the asset holding gridpoints for distribution
    λ::Float64=0.0                   # multiplier
)
    """
    contruct an immutable object containg all paramters
    """

    # permanent endowment shock
    e1_grid, e1_Γ = adda_cooper(e1_size, 0.0, e1_σ)
    e1_Γ = e1_Γ[1, :]
    # e1_grid = [-e1_σ, e1_σ]
    # e1_Γ = Matrix(1.0I, e1_size, e1_size)
    # G_e1 = [1.0 / e1_size for i = 1:e1_size]
    G_e1 = e1_Γ

    # persistent endowment shock
    inv_e2_σ = 1.0 / e2_σ
    e2_MC = tauchen(e2_size, e2_ρ, e2_σ, 0.0, 3)
    # e2_MC = rouwenhorst(e2_size, e2_ρ, e2_σ, 0.0)
    e2_Γ = e2_MC.p
    e2_grid = collect(e2_MC.state_values)
    # e2_grid, e2_Γ = adda_cooper(e2_size, e2_ρ, e2_σ)
    G_e2 = stationary_distributions(MarkovChain(e2_Γ, e2_grid))[1]
    # G_e2 = [1.0, 0.0, 0.0]

    # transitory endowment shock
    e3_grid, e3_Γ = adda_cooper(e3_size, 0.0, e3_σ)
    e3_Γ = e3_Γ[1, :]
    # e3_bar = sqrt((3 / 2) * e3_σ^2)
    # e3_grid = [-e3_bar, 0.0, e3_bar]
    # e3_Γ = [1.0 / e3_size for i = 1:e3_size]
    G_e3 = e3_Γ # [0.0, 1.0, 0.0]

    # aggregate labor endowment
    E = 1.0

    # preference schock
    ν_grid = ones(ν_size)
    ν_p_1 = 0.98
    ν_Γ = [ν_p_1, 1.0 - ν_p_1]
    G_ν = ν_Γ

    # asset holding grid for VFI
    a_min = -1.5 * exp(e1_grid[end] + e2_grid[end] + e3_grid[end])
    a_grid_neg = ((range(a_size_neg - 1, stop=0.0, length=a_size_neg) / (a_size_neg - 1)) .^ a_degree) * a_min
    a_grid_neg = a_grid_neg[1:(end-1)]
    a_grid_pos = ((range(0.0, stop=a_size_pos - 1, length=a_size_pos) / (a_size_pos - 1)) .^ a_degree) * a_max
    a_grid = cat(a_grid_neg, a_grid_pos, dims=1)
    a_size = length(a_grid)
    a_ind_zero = a_size_neg
    a_size_neg = a_size_neg - 1

    # asset holding grid for μ
    a_size_neg_μ = a_size_neg * μ_scale
    a_size_pos_μ = a_size_pos * μ_scale
    a_grid_neg_μ = collect(range(a_min, 0.0, length=a_size_neg_μ))
    a_grid_pos_μ = collect(range(0.0, a_max, length=a_size_pos_μ))
    # a_grid_pos_μ = ((range(0.0, stop=a_size_pos_μ - 1, length=a_size_pos_μ) / (a_size_pos_μ - 1)) .^ a_degree) * a_max
    a_grid_μ = cat(a_grid_neg_μ[1:(end-1)], a_grid_pos_μ, dims=1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(iszero, a_grid_μ)[]

    # aggregate prices 
    ξ_λ = (1.0 - ψ) / (1.0 - λ - ψ)
    Λ_λ = β_f * (1.0 - ψ + ψ * ξ_λ)
    LR_λ = ξ_λ / θ
    KL2D_λ = LR_λ / (LR_λ - 1.0)
    ι_λ = λ * θ / Λ_λ
    r_k_λ = r_f + ι_λ
    K_λ = E * ((r_k_λ + δ) / α)^(1.0 / (α - 1.0))
    w_λ = (1.0 - α) * (K_λ / E)^α

    # iterators
    # loop_V = collect(Iterators.product(1:ν_size, 1:e3_size, 1:e2_size, 1:e1_size, 1:a_size))
    loop_e2_e1 = CartesianIndices((e2_size, e1_size))
    loop_a_neg_e2_e1 = CartesianIndices((a_size_neg, e2_size, e1_size))
    loop_ν_e2_e1 = CartesianIndices((ν_size, e2_size, e1_size))
    loop_a_ν_e2_e1 = CartesianIndices((a_size, ν_size, e2_size, e1_size))
    loop_a_neg_ν_e2_e1 = CartesianIndices((a_size_neg, ν_size, e2_size, e1_size))
    loop_a_pos_ν_e2_e1 = CartesianIndices((a_size_pos, ν_size, e2_size, e1_size))
    loop_e3_ν_e2_e1 = CartesianIndices((e3_size, ν_size, e2_size, e1_size))
    loop_a_neg_e3_ν_e1 = CartesianIndices((a_size_neg, e3_size, ν_size, e1_size))

    # dsicounted aggregate shock transition
    Γ = zeros(e3_size, ν_size, e2_size, ν_size, e2_size)
    ρβν = ρ * β * ν_grid
    for e2_i in 1:e2_size, ν_i in 1:ν_size, e2_p_i in 1:e2_size, ν_p_i in 1:ν_size, e3_p_i in 1:e3_size
        Γ[e3_p_i, ν_p_i, e2_p_i, ν_i, e2_i] = ρβν[ν_i] * e3_Γ[e3_p_i] * ν_Γ[ν_p_i] * e2_Γ[e2_i, e2_p_i]
    end
    Γ_e3_ν = zeros(e3_size, ν_size)
    for ν_p_i in 1:ν_size, e3_p_i in 1:e3_size
        Γ_e3_ν[e3_p_i, ν_p_i] = e3_Γ[e3_p_i] * ν_Γ[ν_p_i]
    end

    # precomputation of handy scalars and matrices
    R_bar = ρ ./ ((-a_grid_neg) .* (1.0 + r_f + τ + ι_λ))
    q_bar = ρ / (1.0 + r_f)

    W = zeros(e3_size, e2_size, e1_size)
    WA = zeros(a_size, e3_size, e2_size, e1_size)
    u_d = zeros(e3_size, e2_size, e1_size)
    for e1_i = 1:e1_size, e2_i = 1:e2_size, e3_i = 1:e3_size
        e1 = e1_grid[e1_i]
        e2 = e2_grid[e2_i]
        e3 = e3_grid[e3_i]
        W_temp = w_λ * exp(e1 + e2 + e3)
        W[e3_i, e2_i, e1_i] = W_temp
        WA[:, e3_i, e2_i, e1_i] .= W_temp .+ a_grid
        u_d[e3_i, e2_i, e1_i] = utility((1.0 - η) * W_temp - κ, γ)
    end

    e2_μ_grid = e2_ρ .* e2_grid
    e2_μ_σ2_grid = e2_μ_grid .+ e2_σ^2.0
    e2_μ_σ2_b2_grid = e2_μ_σ2_grid ./ 2.0
    Γ_default = zeros(e3_size, e2_size, e1_size)
    for e1_i = 1:e1_size, e2_i = 1:e2_size, e3_i = 1:e3_size
        e1 = e1_grid[e1_i]
        e2_μ_σ2_b2 = e2_μ_σ2_b2_grid[e2_i]
        e3 = e3_grid[e3_i]
        Γ_default[e3_i, e2_i, e1_i] = η * w_λ * exp(e1 + e3) * exp(e2_μ_σ2_b2)
    end

    # return values
    return (
        β=β,
        ρ=ρ,
        r_f=r_f,
        β_f=β_f,
        τ=τ,
        γ=γ,
        δ=δ,
        α=α,
        ψ=ψ,
        θ=θ,
        Ph=Ph,
        η=η,
        ξ=ξ,
        κ=κ,
        e1_σ=e1_σ,
        e1_size=e1_size,
        e1_Γ=e1_Γ,
        e1_grid=e1_grid,
        G_e1=G_e1,
        e2_ρ=e2_ρ,
        e2_σ=e2_σ,
        inv_e2_σ=inv_e2_σ,
        e2_size=e2_size,
        e2_Γ=e2_Γ,
        e2_grid=e2_grid,
        G_e2=G_e2,
        e3_σ=e3_σ,
        e3_size=e3_size,
        e3_Γ=e3_Γ,
        e3_grid=e3_grid,
        G_e3=G_e3,
        E=E,
        ν_size=ν_size,
        ν_Γ=ν_Γ,
        ν_grid=ν_grid,
        G_ν=G_ν,
        a_min=a_min,
        a_max=a_max,
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
        λ=λ,
        ξ_λ=ξ_λ,
        Λ_λ=Λ_λ,
        LR_λ=LR_λ,
        KL2D_λ=KL2D_λ,
        ι_λ=ι_λ,
        r_k_λ=r_k_λ,
        K_λ=K_λ,
        w_λ=w_λ,
        loop_e2_e1=loop_e2_e1,
        loop_a_neg_e2_e1=loop_a_neg_e2_e1,
        loop_ν_e2_e1=loop_ν_e2_e1,
        loop_a_ν_e2_e1=loop_a_ν_e2_e1,
        loop_a_neg_ν_e2_e1=loop_a_neg_ν_e2_e1,
        loop_a_pos_ν_e2_e1=loop_a_pos_ν_e2_e1,
        loop_e3_ν_e2_e1=loop_e3_ν_e2_e1,
        loop_a_neg_e3_ν_e1=loop_a_neg_e3_ν_e1,
        Γ=Γ,
        Γ_e3_ν=Γ_e3_ν,
        R_bar=R_bar,
        q_bar=q_bar,
        W=W,
        WA=WA,
        u_d=u_d,
        e2_μ_grid=e2_μ_grid,
        e2_μ_σ2_grid=e2_μ_σ2_grid,
        e2_μ_σ2_b2_grid=e2_μ_σ2_b2_grid,
        Γ_default=Γ_default,
    )
end

@inline function find_min_bounds(obj, grid_min::Real, grid_max::Real; grid_length::Int=120, neighborhood::Int=1)
    @assert grid_max > grid_min
    @assert grid_length ≥ 2
    @assert neighborhood ≥ 1

    step = (grid_max - grid_min) / (grid_length - 1)

    best_val = Inf
    best_i = 1
    @inbounds @simd for i in 1:grid_length
        x = grid_min + (i - 1) * step
        y = obj(x)
        if isfinite(y) && y < best_val
            best_val = y
            best_i = i
        end
    end

    lo_i = max(1, best_i - neighborhood)
    hi_i = min(grid_length, best_i + neighborhood)

    lb = grid_min + (lo_i - 1) * step
    ub = grid_min + (hi_i - 1) * step
    if lb == ub
        ub = min(grid_max, lb + step)
        lb = max(grid_min, ub - step)
    end
    return lb, ub
end

function zero_bounds_function(V_d::Float64, V_nd::Vector{Float64}, a_grid::Vector{Float64})
    """
    compute bounds for (zero) root finding
    """
    @inbounds lb = a_grid[findlast(V_nd .< V_d)]
    @inbounds ub = a_grid[findfirst(V_nd .> V_d)]
    return lb, ub
end

function utility(c::Float64, γ::Float64)
    """
    compute utility of CRRA utility function with coefficient γ
    """
    if c > 0.0
        return γ == 1.0 ? log(c) : 1.0 / ((1.0 - γ) * c^(γ - 1.0))
    else
        return -Inf
    end
end

function inverse_utility(u::Float64, γ::Float64)
    """
    compute inverse utility of CRRA utility function with coefficient γ
    """
    if u == -Inf
        return 0.0
    else
        if γ == 1.0
            return exp(u)
        else
            denominator = (1.0 - γ) * u
            if denominator > 0.0
                return denominator^(1.0 / (1.0 - γ))
            else
                return 0.0
            end
        end
    end
end

# @inline @inbounds function repayment(thres_e2::Float64, a_p_i::Int64, e3_p_i::Int64, e2_i::Int64, e1_i::Int64, parameters::NamedTuple; wage_garnishment::Bool=true)::Float64
#     """
#     evaluate repayment analytically with and without wage garnishment
#     """
    
#     @unpack a_grid_neg, inv_e2_σ, e2_μ_σ2_grid, Γ_default = parameters
    
#     e2_μ_σ2 = e2_μ_σ2_grid[e2_i]
#     default_prob = normcdf((thres_e2 - e2_μ_σ2) * inv_e2_σ)
#     a_p = a_grid_neg[a_p_i]
#     total_amount = -a_p * (1.0 - default_prob)
#     wage_garnishment && (total_amount += Γ_default[e3_p_i, e2_i, e1_i] * default_prob)
    
#     return clamp(total_amount, 0.0, -a_p)
# end

# @inline @views @inbounds 
function repayment_mat(thres_e2::AbstractArray{Float64,2}, a_p_i::Int64, e2_i::Int64, e1_i::Int64, parameters::NamedTuple)::Matrix{Float64}
    """
    evaluate repayment analytically with and without wage garnishment
    """
    
    @unpack a_grid_neg, inv_e2_σ, e2_μ_σ2_grid, Γ_default = parameters
    
    e2_μ_σ2 = e2_μ_σ2_grid[e2_i]
    a_p = a_grid_neg[a_p_i]
    Γ_e3 = Γ_default[:, e2_i, e1_i] 
    default_probs = normcdf.((thres_e2 .- e2_μ_σ2) .* inv_e2_σ)
    total_amounts = -a_p .* (1.0 .- default_probs) .+ Γ_e3 .* default_probs
    return clamp.(total_amounts, 0.0, -a_p)
end

mutable struct MutableAggregateVariables{T}
    K::T
    L::T
    L_adj::T
    D::T
    N::T
    profit::T
    ω::T
    LR::T
    KL2D::T
    debt_to_earning_ratio::T
    share_of_filers::T
    share_of_involuntary_filers::T
    share_in_debts::T
    avg_loan_rate::T
    avg_loan_rate_pw::T
end

# 2) Main container: parametric and concrete
mutable struct MutableVariables{T,
    A3<:AbstractArray{T,3},A4<:AbstractArray{T,4},A5<:AbstractArray{T,5},A6<:AbstractArray{T,6}}
    aggregate_variables::MutableAggregateVariables{T}
    R::A3
    q::A3
    rbl_a::AbstractArray{T,2}
    rbl_qa::AbstractArray{T,2}
    V::A5
    V_d::A4
    V_nd::A5
    V_pos::A5
    EV::A4
    EV_pos::A4
    EV_Ph::A4
    policy_a::A5
    policy_d::A5
    policy_a_pos::A5
    thres_a::A4
    thres_e2::A4
    μ::A6
end

@views @inbounds function create_variables(parameters::NamedTuple; T::Type{<:Real}=Float64)

    # ---- Assets / grids / sizes
    @unpack a_min, a_max, a_degree,
    a_size, a_size_neg, a_size_pos,
    a_grid, a_grid_neg, a_grid_pos,
    a_ind_zero,
    a_size_μ, a_size_pos_μ, a_ind_zero_μ = parameters

    # ---- Shocks: sizes, grids, transitions
    @unpack e1_size, e1_grid, e1_Γ,
    e2_size, e2_grid, e2_Γ, e2_ρ, e2_σ,
    e3_size, e3_grid, e3_Γ,
    ν_size, ν_Γ = parameters

    # ---- Prices / policy / model scalars
    @unpack ρ, r_f, τ, η, κ, w_λ,
    R_bar, q_bar, Γ_e3_ν = parameters

    @unpack loop_a_neg_e3_ν_e1, loop_a_neg_e2_e1, loop_e2_e1 = parameters

    # -- Aggregates
    agg = MutableAggregateVariables{T}(
        zero(T), zero(T), zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T), zero(T), zero(T))

    # -- Prices / schedules
    thres_a = Array{T}(undef, e3_size, ν_size, e2_size, e1_size)
    thres_e2 = Array{T}(undef, a_size_neg, e3_size, ν_size, e1_size)

    # --- Fill threshold_e2 with fused broadcasts (no inner scalar loops)
    @batch for idx in loop_a_neg_e3_ν_e1
        a_neg_i, e3_i, ν_i, e1_i = idx.I
        e1 = e1_grid[e1_i]
        e3 = e3_grid[e3_i]
        a_neg = a_grid_neg[a_neg_i]
        thres_e2[a_neg_i, e3_i, ν_i, e1_i] = log_(-a_neg / w_λ) - e1 - e3
    end

    R = Array{T}(undef, a_size_neg, e2_size, e1_size)
    q = fill(q_bar, a_size, e2_size, e1_size)

    # --- Compute R and q; find rbl via bounded 1d optimize
    @batch for idx in loop_a_neg_e2_e1
        a_p_i, e2_i, e1_i = idx.I        
        thres_e2_ = thres_e2[a_p_i, :, :, e1_i]
        repayment_e3_ν = repayment_mat(thres_e2_, a_p_i, e2_i, e1_i, parameters)
        R_temp = sum(Γ_e3_ν .* repayment_e3_ν)
        R[a_p_i, e2_i, e1_i] = R_temp
        q[a_p_i, e2_i, e1_i] = R_bar[a_p_i] * R_temp
    end

    rbl_a = Array{T}(undef, e2_size, e1_size)
    rbl_qa = Array{T}(undef, e2_size, e1_size)

    @batch for idx in loop_e2_e1
        e2_i, e1_i = idx.I        
        q_grid_neg = q[1:a_size_neg, e2_i, e1_i]
        rbl_a_, rbl_qa_, _ = find_min_qa(a_grid_neg, q_grid_neg)
        rbl_a[e2_i, e1_i] = rbl_a_
        rbl_qa[e2_i, e1_i] = rbl_qa_
    end

    # -- Value functions (undef if you fully set later; zeros if used before fill)
    V = zeros(T, a_size, e3_size, ν_size, e2_size, e1_size)
    V_d = Array{T}(undef, e3_size, ν_size, e2_size, e1_size)
    V_nd = Array{T}(undef, a_size, e3_size, ν_size, e2_size, e1_size)
    V_pos = zeros(T, a_size_pos, e3_size, ν_size, e2_size, e1_size)
    EV = zeros(T, a_size, ν_size, e2_size, e1_size)
    EV_pos = zeros(T, a_size_pos, ν_size, e2_size, e1_size)
    EV_Ph = zeros(T, a_size_pos, ν_size, e2_size, e1_size)

    # -- Distribution
    μ = zeros(T, a_size_μ, e1_size, e2_size, e3_size, ν_size, 2)
    μ_size = (a_size_μ + a_size_pos_μ) * e1_size * e2_size * e3_size * ν_size
    μ[:, :, :, :, :, 1] .= inv(T(μ_size))
    μ[a_ind_zero_μ:end, :, :, :, :, 2] .= inv(T(μ_size))

    # -- Policies
    policy_a = Array{T}(undef, a_size, e3_size, ν_size, e2_size, e1_size)
    policy_d = Array{T}(undef, a_size, e3_size, ν_size, e2_size, e1_size)
    policy_a_pos = Array{T}(undef, a_size_pos, e3_size, ν_size, e2_size, e1_size)

    return MutableVariables{T,
        typeof(R),typeof(V_d),typeof(V),typeof(μ)}(
        agg, R, q, rbl_a, rbl_qa, V, V_d, V_nd, V_pos, EV, EV_pos, EV_Ph,
        policy_a, policy_d, policy_a_pos, thres_a, thres_e2, μ
    )
end

struct ItpCache{ItpQ,ItpEv,ItpEvPh}
    q::Array{ItpQ,2}                # size: (e2_size, e1_size)
    EV::Array{ItpEv,3}              # size: (ν_size, e2_size, e1_size)
    EV_Ph::Array{ItpEvPh,3}           # size: (ν_size, e2_size, e1_size)
end

@inline _build_itp(xs, ys) = linear_interpolation(xs, ys, extrapolation_bc=Line())

@views @inbounds function build_itp_cache(variables::MutableVariables, parameters::NamedTuple)
    """
    construct the cached interpolants
    """

    @unpack a_grid, a_grid_pos, a_ind_zero, Ph, e1_size, e2_size, ν_size = parameters

    q_ = variables.q[:, 1, 1]
    q_sample = linear_interpolation(a_grid, q_, extrapolation_bc=Line())
    EV_ = variables.EV[:, 1, 1, 1]
    EV_sample = linear_interpolation(a_grid, EV_, extrapolation_bc=Line())
    EV_Ph_ = variables.EV_Ph[:, 1, 1, 1]
    EV_Ph_sample = linear_interpolation(a_grid_pos, EV_Ph_, extrapolation_bc=Line())

    q_itp = Array{typeof(q_sample)}(undef, e2_size, e1_size)
    EV_itp = Array{typeof(EV_sample)}(undef, ν_size, e2_size, e1_size)
    EV_Ph_itp = Array{typeof(EV_Ph_sample)}(undef, ν_size, e2_size, e1_size)

    for e2_i in 1:e2_size, e1_i in 1:e1_size
        q_ = variables.q[:, e2_i, e1_i]
        q_itp[e2_i, e1_i] = linear_interpolation(a_grid, q_, extrapolation_bc=Line())
        for ν_i in 1:ν_size
            EV_ = variables.EV[:, ν_i, e2_i, e1_i]
            EV_itp[ν_i, e2_i, e1_i] = linear_interpolation(a_grid, EV_, extrapolation_bc=Line())
            EV_Ph_ = variables.EV_Ph[:, ν_i, e2_i, e1_i]
            EV_Ph_itp[ν_i, e2_i, e1_i] = linear_interpolation(a_grid_pos, EV_Ph_, extrapolation_bc=Line())
        end
    end

    return ItpCache{typeof(q_sample),typeof(EV_sample),typeof(EV_Ph_sample)}(q_itp, EV_itp, EV_Ph_itp)
end

@views @inbounds function update_EV!(V_p::Array{Float64,5}, V_pos_p::Array{Float64,5}, variables::MutableVariables, parameters::NamedTuple)
    """
    Construct expected value functions `EV` and `EV_pos`
    """

    @unpack a_ind_zero, Ph, Γ, loop_a_ν_e2_e1, loop_a_pos_ν_e2_e1 = parameters

    @batch for idx in loop_a_ν_e2_e1
        a_p_i, ν_i, e2_i, e1_i = idx.I
        Γ_temp = Γ[:, :, :, ν_i, e2_i]
        V_p_temp = V_p[a_p_i, :, :, :, e1_i]
        variables.EV[a_p_i, ν_i, e2_i, e1_i] = sum(Γ_temp .* V_p_temp)
    end

    @batch for idx in loop_a_pos_ν_e2_e1
        a_pos_p_i, ν_i, e2_i, e1_i = idx.I
        a_p_i = a_pos_p_i + a_ind_zero - 1
        EV_temp = variables.EV[a_p_i, ν_i, e2_i, e1_i]
        Γ_temp = Γ[:, :, :, ν_i, e2_i]
        V_pos_p_temp = V_pos_p[a_pos_p_i, :, :, :, e1_i]
        EV_pos_temp = sum(Γ_temp .* V_pos_p_temp)
        variables.EV_pos[a_pos_p_i, ν_i, e2_i, e1_i] = EV_pos_temp
        variables.EV_Ph[a_pos_p_i, ν_i, e2_i, e1_i] = Ph * EV_temp + (1.0 - Ph) * EV_pos_temp
    end

    return nothing
end

function update_V_d!(variables::MutableVariables, parameters::NamedTuple)
    """
    Update the default value function `V_d`
    """

    # @unpack loop_ν_e2_e1, u_d, ξ = parameters
    # @views @inbounds @batch for idx in loop_ν_e2_e1
    #     ν_i, e2_i, e1_i = idx.I
    #     EV_pos_zero = variables.EV_pos[1, ν_i, e2_i, e1_i]
    #     u_d_temp = u_d[:, e2_i, e1_i]
    #     @. variables.V_d[:, ν_i, e2_i, e1_i] = u_d_temp - ξ + EV_pos_zero
    # end

    @unpack loop_e3_ν_e2_e1, u_d, ξ = parameters
    @inbounds @batch for idx in loop_e3_ν_e2_e1
        e3_i, ν_i, e2_i, e1_i = idx.I
        EV_pos_zero = variables.EV_pos[1, ν_i, e2_i, e1_i]
        u_d_temp = u_d[e3_i, e2_i, e1_i]
        variables.V_d[e3_i, ν_i, e2_i, e1_i] = u_d_temp - ξ + EV_pos_zero
    end

    return nothing
end

struct DP_Problem{ItpQ,ItpEv,FunU,T}
    qa_itp::ItpQ
    EV_itp::ItpEv
    utility::FunU
    γ::T
end

@inline function obj_DP(DP::DP_Problem, a_p::Float64, a::Float64, W_::Float64)
    c = W_ + a - DP.qa_itp(a_p)
    return -(DP.utility(c, DP.γ) + DP.EV_itp(a_p))
end

@inline function solve_DP(
    DP::DP_Problem, a::Float64, W_::Float64;
    lb::Float64, ub::Float64,
    rtol::Float64=1e-8, atol::Float64=1e-10, iters::Int=200
)
    if !(ub > lb) || isapprox(ub, lb; rtol=0.0, atol=atol)
        a_star = lb
        v_nd = -obj_DP(DP, a_star, a, W_)
        return v_nd, a_star, 1 # degenerate
    end

    F = (a_p::Float64) -> obj_DP(DP, a_p, a, W_)
    res = Optim.optimize(F, lb, ub, Optim.Brent();
        rel_tol=rtol, abs_tol=atol, iterations=iters)

    if Optim.converged(res) && isfinite(Optim.minimum(res))
        a_star = Optim.minimizer(res)
        v_nd = -Optim.minimum(res)
        return v_nd, a_star, 2 # convergence
    else
        FL = F(lb)
        FU = F(ub)
        if FL <= FU
            return -FL, lb, 3 # boundary
        else
            return -FU, ub, 3 # boundary
        end
    end
end

struct QaInterpolant{Itp}
    q_itp::Itp
end

@inline (f::QaInterpolant)(a_p::Real) = f.q_itp(a_p) * a_p

@inline @inbounds function find_min_qa(a_grid_neg::AbstractVector{T}, q_grid_neg::AbstractVector{T}) where {T<:AbstractFloat}

    Na = length(a_grid_neg)
    @assert Na == length(q_grid_neg) "length mismatch"
    @assert Na ≥ 2 "need at least 2 points"

    best_f, best_a, best_i = typemax(Float64), a_grid_neg[1], 1

    for i in 1:(Na-1)

        a0, a1 = a_grid_neg[i], a_grid_neg[i+1]
        q0, q1 = q_grid_neg[i], q_grid_neg[i+1]

        Da = a1 - a0
        @assert Da > 0 "grid must be strictly ascending at i=$i: a0=$a0, a1=$a1"

        Dq = q1 - q0
        m = Dq / Da

        if m != 0.0
            a_star = (a0 - q0 / m) / 2.0
            if a0 ≤ a_star ≤ a1
                f_star = a_star * (q0 + m * (a_star - a0))
                if f_star < best_f
                    best_f, best_a, best_i = f_star, a_star, i
                end
            end
        end

        f0 = a0 * q0
        if f0 < best_f
            best_f, best_a, i = f0, a0, i
        end

        f1 = a1 * q1
        if f1 < best_f
            best_f, best_a, i = f1, a1, i
        end
    end

    return best_a, best_f, best_i
end

@views @inbounds function update_value_and_policy_functions!(
    V_p::Array{Float64,5},
    V_pos_p::Array{Float64,5},
    variables::MutableVariables,
    parameters::NamedTuple,
    itp_cache::ItpCache
)
    """
    one-step update of value and policy functions
    """

    @unpack a_size, a_grid, a_size_pos, a_grid_pos, a_ind_zero, a_min = parameters
    @unpack e1_size, e1_grid, e1_Γ, e2_size, e2_grid, e2_Γ, e3_size, e3_grid, e3_Γ = parameters
    @unpack ν_size, ν_grid, ν_Γ = parameters
    @unpack ρ, β, γ, r_f = parameters
    @unpack Ph, η, κ, ξ, W, q_bar, loop_e2_e1 = parameters

    update_EV!(V_p, V_pos_p, variables, parameters)
    update_V_d!(variables, parameters)

    # @batch 
    for idx in loop_e2_e1

        e2_i, e1_i = idx.I

        q_ = variables.q[:, e2_i, e1_i]
        q_itp = itp_cache.q[e2_i, e1_i]
        copyto!(q_itp.itp.coefs, q_)
        qa_itp = QaInterpolant(q_itp)

        rbl_a_ = variables.rbl_a[e2_i, e1_i]
        rbl_qa_ = variables.rbl_qa[e2_i, e1_i]

        for ν_i = 1:ν_size

            EV_ = variables.EV[:, ν_i, e2_i, e1_i]
            EV_itp = itp_cache.EV[ν_i, e2_i, e1_i]
            copyto!(EV_itp.itp.coefs, EV_)

            EV_Ph_ = variables.EV_Ph[:, ν_i, e2_i, e1_i]
            EV_Ph_itp = itp_cache.EV_Ph[ν_i, e2_i, e1_i]
            copyto!(EV_Ph_itp.itp.coefs, EV_Ph_)

            DP_Problem_nd = DP_Problem(qa_itp, EV_itp, utility, γ)
            DP_Problem_pos = DP_Problem(qa_itp, EV_Ph_itp, utility, γ)

            for e3_i = 1:e3_size

                W_ = W[e3_i, e2_i, e1_i]
                V_d_ = variables.V_d[e3_i, ν_i, e2_i, e1_i]

                lb_nd = rbl_a_
                lb_pos = 0.0

                for a_i = 1:a_size

                    a = a_grid[a_i]
                    CoH = W_ + a
                    ub_q = CoH / q_bar

                    if ((CoH - rbl_qa_) <= 0.0) || (ub_q <= lb_nd)
                        variables.V_nd[a_i, e3_i, ν_i, e2_i, e1_i] = -Inf
                        variables.V[a_i, e3_i, ν_i, e2_i, e1_i] = V_d_
                        variables.policy_a[a_i, e3_i, ν_i, e2_i, e1_i] = 0.0
                        variables.policy_d[a_i, e3_i, ν_i, e2_i, e1_i] = 1.0
                    else
                        V_nd_, a_star_nd, status_nd = solve_DP(DP_Problem_nd, a, W_; lb=lb_nd, ub=ub_q)
                        variables.V_nd[a_i, e3_i, ν_i, e2_i, e1_i] = V_nd_
                        if V_nd_ > V_d_
                            variables.V[a_i, e3_i, ν_i, e2_i, e1_i] = V_nd_
                            variables.policy_a[a_i, e3_i, ν_i, e2_i, e1_i] = a_star_nd
                            variables.policy_d[a_i, e3_i, ν_i, e2_i, e1_i] = 0.0
                        else
                            variables.V[a_i, e3_i, ν_i, e2_i, e1_i] = V_d_
                            variables.policy_a[a_i, e3_i, ν_i, e2_i, e1_i] = 0.0
                            variables.policy_d[a_i, e3_i, ν_i, e2_i, e1_i] = 1.0
                        end
                        if status_nd == 2
                            lb_nd = a_star_nd
                        end
                    end

                    if a_i >= a_ind_zero
                        a_pos_i = a_i - a_ind_zero + 1
                        V_pos_, a_star_pos, status_pos = solve_DP(DP_Problem_pos, a, W_; lb=lb_pos, ub=ub_q)
                        variables.V_pos[a_pos_i, e3_i, ν_i, e2_i, e1_i] = V_pos_
                        variables.policy_a_pos[a_pos_i, e3_i, ν_i, e2_i, e1_i] = a_star_pos
                        if status_pos == 2
                            lb_pos = a_star_pos
                        end
                    end
                end
            end
        end
    end
    return nothing
end

@inline log_(thres_e::Float64) = thres_e > 0.0 ? log(thres_e) : -Inf

@inline function compute_e2_star(W_fc_0::Float64, W_fc_1::Float64, thres_a_fc_0::Float64, thres_a_fc_1::Float64, 
                                 a_neg_::Float64, crossing_idx::Union{Nothing,Int64})::Float64

    m_fc = (thres_a_fc_1 - thres_a_fc_0) / (W_fc_1 - W_fc_0)
    
    if isnothing(crossing_idx)
        # Beyond upper bound - extrapolate
        e2_star = W_fc_1 + (a_neg_ - thres_a_fc_1) / m_fc
    elseif crossing_idx == 1
        # Beyond lower bound - extrapolate  
        e2_star = W_fc_0 - (thres_a_fc_0 - a_neg_) / m_fc
    else
        # Normal interpolation case
        e2_star = W_fc_1 - (thres_a_fc_1 - a_neg_) / m_fc
    end
    
    return e2_star
end

function find_thresholds!(variables::MutableVariables, parameters::NamedTuple; indIU::Bool=true, indE::Bool=true)
    """
    update default thresholds in assets and persistent endowments (e2)
    """

    @unpack a_size_neg, a_grid_neg, e2_size, e2_grid, loop_e3_ν_e2_e1, loop_a_neg_e3_ν_e1 = parameters
    @unpack γ, e1_grid, e3_grid, W, w_λ = parameters

    @inbounds @views @batch for idx in loop_e3_ν_e2_e1

        e3_i, ν_i, e2_i, e1_i = idx.I

        V_d_ = variables.V_d[e3_i, ν_i, e2_i, e1_i]
        V_nd_ = variables.V_nd[:, e3_i, ν_i, e2_i, e1_i]

        first_finite = findfirst(isfinite, V_nd_)
        @assert !isnothing(first_finite) "no finite V_nd exists for (e3,ν,e2,e1) = ($e3_i,$ν_i,$e2_i,$e1_i)"
        @assert first_finite != a_size_neg "finite V_nd exists only at the boundary for (e3,ν,e2,e1) = ($e3_i,$ν_i,$e2_i,$e1_i)"

        V_nd_ff_0 = V_nd_[first_finite]

        if V_nd_ff_0 > V_d_

            first_finite_1 = first_finite + 1
            a_ff_0, a_ff_1 = a_grid_neg[first_finite], a_grid_neg[first_finite_1]
            V_nd_ff_1 = V_nd_[first_finite_1]

            if indIU
                V_d_ = inverse_utility(V_d_, γ)
                V_nd_ff_0 = inverse_utility(V_nd_ff_0, γ)
                V_nd_ff_1 = inverse_utility(V_nd_ff_1, γ)
            end

            m_ff = (V_nd_ff_1 - V_nd_ff_0) / (a_ff_1 - a_ff_0)
            a_star = a_ff_0 - (V_nd_ff_0 - V_d_) / m_ff

        elseif V_nd_ff_0 == V_d_

            a_star = a_grid_neg[first_finite]

        else

            crossing_idx = findfirst(i -> V_nd_[i] > V_d_, first_finite:a_size_neg)
            @assert !isnothing(crossing_idx) "no crossing found: V_nd ≤ V_d for (e3,ν,e2,e1) = ($e3_i,$ν_i,$e2_i,$e1_i)"

            first_cross_1 = first_finite + crossing_idx - 1
            first_cross_0 = first_cross_1 - 1

            a_fc_0, a_fc_1 = a_grid_neg[first_cross_0], a_grid_neg[first_cross_1]
            V_nd_fc_0, V_nd_fc_1 = V_nd_[first_cross_0], V_nd_[first_cross_1]

            if indIU
                V_d_ = inverse_utility(V_d_, γ)
                V_nd_fc_0 = inverse_utility(V_nd_fc_0, γ)
                V_nd_fc_1 = inverse_utility(V_nd_fc_1, γ)
            end

            m_fc = (V_nd_fc_1 - V_nd_fc_0) / (a_fc_1 - a_fc_0)
            a_star = a_fc_1 - (V_nd_fc_1 - V_d_) / m_fc # a_star = a_fc_0 + (V_d_ - V_nd_fc_0) / m_fc
        end

        variables.thres_a[e3_i, ν_i, e2_i, e1_i] = a_star
    end

    @inbounds @views @batch for idx in loop_a_neg_e3_ν_e1

        a_neg_i, e3_i, ν_i, e1_i = idx.I

        a_neg_ = a_grid_neg[a_neg_i]
        thres_a_ = variables.thres_a[e3_i, ν_i, :, e1_i]

        if indE
            W_ = W[e3_i, :, e1_i]
            e3_, e1_ =  e3_grid[e3_i], e1_grid[e1_i]
        else
            W_ = e2_grid
        end

        crossing_idx = findfirst(i -> a_neg_ > thres_a_[i], 1:e2_size)
        
        if isnothing(crossing_idx)
            first_cross_0, first_cross_1 = e2_size - 1, e2_size
        elseif crossing_idx == 1
            first_cross_0, first_cross_1 = 1, 2
        else
            first_cross_0, first_cross_1 = crossing_idx - 1, crossing_idx
        end

        W_fc_0, W_fc_1 = W_[first_cross_0], W_[first_cross_1]
        thres_a_fc_0, thres_a_fc_1 = thres_a_[first_cross_0], thres_a_[first_cross_1]

        e2_star = compute_e2_star(W_fc_0, W_fc_1, thres_a_fc_0, thres_a_fc_1, a_neg_, crossing_idx)

        if indE
            e2_star = log_(e2_star / w_λ) - e3_ - e1_
        end

        variables.thres_e2[a_neg_i, e3_i, ν_i, e1_i] = e2_star
    end

    return nothing
end

@views @inbounds function update_pricing_and_rbl_function!(variables::MutableVariables, parameters::NamedTuple)
    """
    update discounted borrowing price and borrowing risky limit
    """

    @unpack loop_a_neg_e2_e1, Γ_e3_ν, R_bar, loop_e2_e1, a_size_neg, a_grid_neg = parameters

    @batch for idx in loop_a_neg_e2_e1
        a_p_i, e2_i, e1_i = idx.I
        thres_e2_ = variables.thres_e2[a_p_i, :, :, e1_i]
        repayment_e3_ν = repayment_mat(thres_e2_, a_p_i, e2_i, e1_i, parameters)
        R_temp = sum(Γ_e3_ν .* repayment_e3_ν)
        variables.R[a_p_i, e2_i, e1_i] = R_temp
        variables.q[a_p_i, e2_i, e1_i] = R_bar[a_p_i] * R_temp
    end

    # @batch 
    for idx in loop_e2_e1
        e2_i, e1_i = idx.I
        q_grid_neg = variables.q[1:a_size_neg, e2_i, e1_i]
        rbl_a_, rbl_qa_, _ = find_min_qa(a_grid_neg, q_grid_neg)
        variables.rbl_a[e2_i, e1_i] = rbl_a_
        variables.rbl_qa[e2_i, e1_i] = rbl_qa_
    end

    return nothing
end

function solve_value_and_pricing_function!(variables::MutableVariables, parameters::NamedTuple, itp_cache::ItpCache; 
    tol::Float64=1E-8, iter_max::Int64=1000, slow_updating::Float64=1.0)
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
        update_value_and_policy_functions!(V_p, V_pos_p, variables, parameters, itp_cache)

        # default thresholds
        find_thresholds!(variables, parameters; indIU = true, indE = true)

        # pricing function and borrowing risky limit
        update_pricing_and_rbl_function!(variables, parameters)

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

function variables_function_update!(variables::MutableVariables, parameters::NamedTuple; λ::Float64)
    """
    construct a mutable object containing endogenous variables
    """

    # define aggregate prices
    ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ = aggregate_prices_λ_funtion(parameters; λ=λ)
    variables.aggregate_prices = Mutable_Aggregate_Prices(λ, ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_λ, w_λ)
end

function stationary_distribution_function(μ_p::Array{Float64,6}, policy_a::Array{Float64,5}, threshold_a::Array{Float64,4}, policy_pos_a::Array{Float64,5}, policy_pos_d::Array{Float64,5}, parameters::NamedTuple)
    """
    update stationary distribution
    """

    # unpack parameters
    @unpack e1_size, e1_Γ, G_e1, e2_size, e2_Γ, G_e2, e3_size, e3_Γ, G_e3, ν_size, ν_Γ, G_ν, a_grid, a_grid_pos, a_size_μ, a_grid_μ, a_ind_zero_μ, ρ, Ph = parameters

    # construct container
    μ = zeros(a_size_μ, e1_size, e2_size, e3_size, ν_size, 2)

    for e1_i = 1:e1_size, e2_i = 1:e2_size, e3_i = 1:e3_size, ν_i = 1:ν_size

        # interpolated decision rules
        @inbounds @views policy_a_Non_Inf = findall(policy_a[:, e3_i, e2_i, e1_i, ν_i] .!= -Inf)
        @inbounds policy_a_itp = Akima(a_grid[policy_a_Non_Inf], policy_a[policy_a_Non_Inf, e3_i, e2_i, e1_i, ν_i])
        # @inbounds policy_d_itp = Akima(a_grid, policy_d[:, e3_i, e2_i, e1_i, ν_i])
        @inbounds policy_d_itp(x) = x < threshold_a[e3_i, e2_i, e1_i, ν_i] ? 1.0 : 0.0
        @inbounds policy_pos_a_itp = Akima(a_grid_pos, policy_pos_a[:, e3_i, e2_i, e1_i, ν_i])
        @inbounds policy_pos_d_itp = Akima(a_grid_pos, policy_pos_d[:, e3_i, e2_i, e1_i, ν_i])

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
            for e1_p_i = 1:e1_size, e2_p_i = 1:e2_size, e3_p_i = 1:e3_size, ν_p_i = 1:ν_size
                if e1_p_i == e1_i
                    if policy_d_itp(a_μ) == 0.0
                        @inbounds μ[a_p_lb, e1_i, e2_p_i, e3_p_i, ν_p_i, 1] += ρ * e2_Γ[e2_i, e2_p_i] * e3_Γ[e3_p_i] * ν_Γ[ν_p_i] * weight_lower * μ_p[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1]
                        @inbounds μ[a_p_ub, e1_i, e2_p_i, e3_p_i, ν_p_i, 1] += ρ * e2_Γ[e2_i, e2_p_i] * e3_Γ[e3_p_i] * ν_Γ[ν_p_i] * weight_upper * μ_p[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1]
                    else
                        @inbounds μ[a_ind_zero_μ, e1_i, e2_p_i, e3_p_i, ν_p_i, 2] += ρ * e2_Γ[e2_i, e2_p_i] * e3_Γ[e3_p_i] * ν_Γ[ν_p_i] * μ_p[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1]
                    end
                end
                @inbounds μ[a_ind_zero_μ, e1_p_i, e2_p_i, e3_p_i, ν_p_i, 1] += (1.0 - ρ) * G_e1[e1_p_i] * G_e2[e2_p_i] * G_e3[e3_p_i] * G_ν[ν_p_i] * μ_p[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1]
            end

            if a_μ >= 0.0
                @inbounds a_p = clamp(policy_pos_a_itp(a_μ), 0.0, a_grid[end])
                @inbounds d_p = clamp(policy_pos_d_itp(a_μ), 0.0, 1.0)
                a_p_lb = findall(a_grid_μ .<= a_p)[end]
                a_p_ub = findall(a_p .<= a_grid_μ)[1]
                if a_p_lb != a_p_ub
                    @inbounds a_p_lower = a_grid_μ[a_p_lb]
                    @inbounds a_p_upper = a_grid_μ[a_p_ub]
                    weight_lower = (a_p_upper - a_p) / (a_p_upper - a_p_lower)
                    weight_upper = (a_p - a_p_lower) / (a_p_upper - a_p_lower)
                else
                    weight_lower = 0.5
                    weight_upper = 0.5
                end
                for e1_p_i = 1:e1_size, e2_p_i = 1:e2_size, e3_p_i = 1:e3_size, ν_p_i = 1:ν_size
                    if e1_p_i == e1_i
                        @inbounds μ[a_p_lb, e1_i, e2_p_i, e3_p_i, ν_p_i, 1] += (1.0 - d_p) * ρ * Ph * e2_Γ[e2_i, e2_p_i] * e3_Γ[e3_p_i] * ν_Γ[ν_p_i] * weight_lower * μ_p[a_μ_i, e3_i, e2_i, e1_i, ν_i, 2]
                        @inbounds μ[a_p_ub, e1_i, e2_p_i, e3_p_i, ν_p_i, 1] += (1.0 - d_p) * ρ * Ph * e2_Γ[e2_i, e2_p_i] * e3_Γ[e3_p_i] * ν_Γ[ν_p_i] * weight_upper * μ_p[a_μ_i, e3_i, e2_i, e1_i, ν_i, 2]
                        @inbounds μ[a_p_lb, e1_i, e2_p_i, e3_p_i, ν_p_i, 2] += d_p * ρ * (1.0 - Ph) * e2_Γ[e2_i, e2_p_i] * e3_Γ[e3_p_i] * ν_Γ[ν_p_i] * weight_lower * μ_p[a_μ_i, e3_i, e2_i, e1_i, ν_i, 2]
                        @inbounds μ[a_p_ub, e1_i, e2_p_i, e3_p_i, ν_p_i, 2] += d_p * ρ * (1.0 - Ph) * e2_Γ[e2_i, e2_p_i] * e3_Γ[e3_p_i] * ν_Γ[ν_p_i] * weight_upper * μ_p[a_μ_i, e3_i, e2_i, e1_i, ν_i, 2]
                    end
                    @inbounds μ[a_ind_zero_μ, e1_p_i, e2_p_i, e3_p_i, ν_p_i, 1] += (1.0 - ρ) * G_e1[e1_p_i] * G_e2[e2_p_i] * G_e3[e3_p_i] * G_ν[ν_p_i] * μ_p[a_μ_i, e3_i, e2_i, e1_i, ν_i, 2]
                end
            end
        end
    end

    # standardize distribution
    # sum_μ = sum(μ)
    # println("sum_μ = $sum_μ")
    μ = μ ./ sum(μ)

    # return result
    return μ
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
        variables.μ = stationary_distribution_function(μ_p, variables.policy_a, variables.threshold_a, variables.policy_pos_a, variables.policy_pos_d, parameters)

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

function solve_aggregate_variable_function(
    policy_a::Array{Float64,5},
    threshold_a::Array{Float64,4},
    policy_pos_a::Array{Float64,5},
    policy_pos_d::Array{Float64,5},
    q::Array{Float64,3},
    rbl::Array{Float64,3},
    μ::Array{Float64,6},
    K::Float64,
    w::Float64,
    ι::Float64,
    parameters::NamedTuple,
)
    """
    compute equlibrium aggregate variables
    """

    # unpack parameters
    @unpack e1_size, e1_grid, e2_size, e2_grid, e3_size, e3_grid, ν_size, a_grid, a_grid_neg, a_grid_pos, a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ, a_grid_μ, a_size_μ, r_f, τ, ψ, η = parameters

    # initialize container
    K = K
    L = 0.0
    L_adj = 0.0
    D = 0.0
    N = 0.0
    profit = 0.0
    ω = 0.0
    leverage_ratio = 0.0
    KL_to_D_ratio = 0.0
    debt_to_earning_ratio = 0.0
    debt_to_earning_ratio_num = 0.0
    debt_to_earning_ratio_den = 0.0
    share_of_filers = 0.0
    share_of_involuntary_filers = 0.0
    share_in_debts = 0.0
    avg_loan_rate = 0.0
    avg_loan_rate_num = 0.0
    avg_loan_rate_den = 0.0
    avg_loan_rate_pw = 0.0
    avg_loan_rate_pw_num = 0.0
    avg_loan_rate_pw_den = 0.0

    # total loans, deposits, share of filers, nad debt-to-earning ratio
    for e1_i = 1:e1_size, e2_i = 1:e2_size, e3_i = 1:e3_size, ν_i = 1:ν_size

        # interpolated decision rules
        @inbounds @views policy_a_Non_Inf = findall(policy_a[:, e3_i, e2_i, e1_i, ν_i] .!= -Inf)
        @inbounds policy_a_itp = Akima(a_grid[policy_a_Non_Inf], policy_a[policy_a_Non_Inf, e3_i, e2_i, e1_i, ν_i])
        # @inbounds policy_d_itp = Akima(a_grid, policy_d[:, e3_i, e2_i, e1_i, ν_i])
        @inbounds policy_d_itp(x) = x < threshold_a[e3_i, e2_i, e1_i, ν_i] ? 1.0 : 0.0
        @inbounds policy_pos_a_itp = Akima(a_grid_pos, policy_pos_a[:, e3_i, e2_i, e1_i, ν_i])
        @inbounds policy_pos_d_itp = Akima(a_grid_pos, policy_pos_d[:, e3_i, e2_i, e1_i, ν_i])

        # interpolated discounted borrowing amount
        @inbounds @views q_e = q[:, e1_i, e2_i]
        q_function_itp = Akima(a_grid, q_e)
        qa_function_itp = Akima(a_grid, q_e .* a_grid)

        # loop over the dimension of asset holding
        for a_μ_i = 1:a_size_μ

            # extract wealth and compute asset choice
            @inbounds a_μ = a_grid_μ[a_μ_i]
            @inbounds a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])

            if a_p < 0.0
                # total loans
                @inbounds L += -(μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ)) * qa_function_itp(a_p))

                # average loan rate
                avg_loan_rate_num += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ)) * (1.0 / q_function_itp(a_p) - 1.0)
                avg_loan_rate_den += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ))

                # average loan rate (persons-weighted)
                avg_loan_rate_pw_num += (1.0 - policy_d_itp(a_μ)) * (1.0 / q_function_itp(a_p) - 1.0)
                avg_loan_rate_pw_den += 1
            else
                # total deposits
                if a_p > 0.0
                    @inbounds D += (μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ)) * qa_function_itp(a_p))
                    # @inbounds D += (μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * qa_function_itp(a_p))

                end
            end

            if a_μ >= 0.0
                @inbounds a_pos_p = clamp(policy_pos_a_itp(a_μ), 0.0, a_grid[end])
                @inbounds d_p = clamp(policy_pos_d_itp(a_μ), 0.0, 1.0)
                if a_pos_p > 0.0
                    @inbounds D += (μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 2] * qa_function_itp(a_pos_p))
                end
                @inbounds share_of_filers += (μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 2] * d_p)
            end

            if a_μ < 0.0
                # share of filers
                @inbounds share_of_filers += (μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * policy_d_itp(a_μ))

                # share of involuntary filers
                if w * exp(e1_grid[e1_i] + e2_grid[e2_i] + e3_grid[e3_i]) + a_μ - rbl[e1_i, e2_i, 2] < 0.0
                    @inbounds share_of_involuntary_filers += (μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * policy_d_itp(a_μ))
                end

                # debt-to-earning ratio
                # @inbounds debt_to_earning_ratio += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (-a_μ / (w * exp(e1_grid[e1_i] + e2_grid[e2_i] + e3_grid[e3_i])))
                @inbounds debt_to_earning_ratio_num += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (-a_μ)
                # @inbounds debt_to_earning_ratio_den += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (w * exp(e1_grid[e1_i] + e2_grid[e2_i] + e3_grid[e3_i]))

                # loans returned
                L_adj += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * ((-a_μ) * (1.0 - policy_d_itp(a_μ)) + policy_d_itp(a_μ) * η * w * exp(e1_grid[e1_i] + e2_grid[e2_i] + e3_grid[e3_i]))
            end

            @inbounds debt_to_earning_ratio_den += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (w * exp(e1_grid[e1_i] + e2_grid[e2_i] + e3_grid[e3_i]))
            @inbounds debt_to_earning_ratio_den += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 2] * (w * exp(e1_grid[e1_i] + e2_grid[e2_i] + e3_grid[e3_i]))
        end
    end

    # net worth
    N = (K + L) - D

    # exogenous dividend policy
    # profit = (1.0 + r_f + ι) * K + (1.0 + τ + ι) * L - (1.0 + r_f) * D
    profit = ι * (K + L) + (1.0 + r_f) * N
    # ω = (N - ψ * profit) / ((1.0 - ψ) * profit)
    # ω = N / (ψ * profit)
    # ω = (N - ψ * profit) / ((1.0 - ψ) * (K + L))
    ω = (N - ψ * profit) / (K + L)
    # ω = N - ψ * profit

    # leverage ratio
    leverage_ratio = (K + L) / N

    # capital-loan-to-deposit ratio
    KL_to_D_ratio = (K + L) / D

    # debt-to-earning ratio
    # debt_to_earning_ratio = debt_to_earning_ratio_num / debt_to_earning_ratio_den
    # debt_to_earning_ratio = L / w
    debt_to_earning_ratio = debt_to_earning_ratio_num / w

    # average loan rate
    avg_loan_rate = avg_loan_rate_num / avg_loan_rate_den
    avg_loan_rate_pw = avg_loan_rate_pw_num / avg_loan_rate_pw_den

    # share in debt
    share_in_debts = sum(μ[1:(a_ind_zero_μ-1), :, :, :, :, 1])

    # return results
    aggregate_variables = Mutable_Aggregate_Variables(K, L, L_adj, D, N, profit, ω, leverage_ratio, KL_to_D_ratio, debt_to_earning_ratio, share_of_filers, share_of_involuntary_filers, share_in_debts, avg_loan_rate, avg_loan_rate_pw)
    return aggregate_variables
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
    @unpack e1_size, e1_grid, e2_size, e2_grid, e3_size, e3_grid, ν_size, a_grid, a_grid_neg, a_grid_pos, a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ, a_grid_μ, a_size_μ, r_f, τ, ψ, η = parameters

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
    for e1_i = 1:e1_size, e2_i = 1:e2_size, e3_i = 1:e3_size, ν_i = 1:ν_size

        # interpolated decision rules
        @inbounds @views policy_a_Non_Inf = findall(policy_a[:, e3_i, e2_i, e1_i, ν_i] .!= -Inf)
        @inbounds policy_a_itp = Akima(a_grid[policy_a_Non_Inf], policy_a[policy_a_Non_Inf, e3_i, e2_i, e1_i, ν_i])
        @inbounds policy_d_itp = Akima(a_grid, policy_d[:, e3_i, e2_i, e1_i, ν_i])
        @inbounds policy_pos_a_itp = Akima(a_grid_pos, policy_pos_a[:, e3_i, e2_i, e1_i, ν_i])
        @inbounds policy_pos_d_itp = Akima(a_grid_pos, policy_pos_d[:, e3_i, e2_i, e1_i, ν_i])

        # interpolated discounted borrowing amount
        @inbounds @views q_e = q[:, e1_i, e2_i]
        q_function_itp = Akima(a_grid, q_e)

        # loop over the dimension of asset holding
        for a_μ_i = 1:a_size_μ

            # extract wealth and compute asset choice
            @inbounds a_μ = a_grid_μ[a_μ_i]
            @inbounds a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])

            if a_p < 0.0
                # average loan rate
                avg_loan_rate_num += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ)) * (1.0 / q_function_itp(a_p) - 1.0)
                avg_loan_rate_den += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ))
                if (e1_i == 1) && (e2_i == 2)
                    avg_loan_rate_num_permanent_low += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ)) * (1.0 / q_function_itp(a_p) - 1.0)
                    avg_loan_rate_den_permanent_low += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ))
                end
                if (e1_i == 2) && (e2_i == 2)
                    avg_loan_rate_num_permanent_high += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ)) * (1.0 / q_function_itp(a_p) - 1.0)
                    avg_loan_rate_den_permanent_high += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (1.0 - policy_d_itp(a_μ))
                end
            end

            if a_μ < 0.0
                # share of filers
                @inbounds share_of_filers += (μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * policy_d_itp(a_μ))
                if (e1_i == 1) && (e2_i == 2)
                    share_of_filers_permanent_low += (μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * policy_d_itp(a_μ)) / sum(μ[:, e1_i, e2_i, :, :, :])
                end
                if (e1_i == 2) && (e2_i == 2)
                    share_of_filers_permanent_high += (μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * policy_d_itp(a_μ)) / sum(μ[:, e1_i, e2_i, :, :, :])
                end

                # debt-to-earning ratio
                @inbounds debt_to_earning_ratio_num += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (-a_μ)
                if (e1_i == 1) && (e2_i == 2)
                    @inbounds debt_to_earning_ratio_num_permanent_low += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (-a_μ) / sum(μ[:, e1_i, e2_i, :, :, :])
                end
                if (e1_i == 2) && (e2_i == 2)
                    @inbounds debt_to_earning_ratio_num_permanent_high += μ[a_μ_i, e3_i, e2_i, e1_i, ν_i, 1] * (-a_μ) / sum(μ[:, e1_i, e2_i, :, :, :])
                end
            end
        end
    end

    # debt-to-earning ratio
    debt_to_earning_ratio = debt_to_earning_ratio_num / w
    debt_to_earning_ratio_permanent_low = debt_to_earning_ratio_num_permanent_low / (w * exp(e1_grid[1]))
    debt_to_earning_ratio_permanent_high = debt_to_earning_ratio_num_permanent_high / (w * exp(e1_grid[2]))

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
