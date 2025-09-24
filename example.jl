using Distributions, StatsFuns, QuadGK
using LinearAlgebra
using Optim
using Parameters: @unpack
using PrettyTables
using ProgressMeter
using QuantEcon: rouwenhorst, tauchen, stationary_distributions, MarkovChain
using Plots
using Random123
using BenchmarkTools, Profile
using Polyester
using Interpolations

function adda_cooper(N::Integer, ρ::T, σ::T; μ::T=zero(T), rtol::Real=1e-10, atol::Real=0.0) where {T<:AbstractFloat}

    N ≥ 2 || throw(ArgumentError("N ≥ 2 required"))
    abs(ρ) < one(T) || throw(ArgumentError("|ρ| < 1 required for stationarity"))
    σ > zero(T) || throw(ArgumentError("σ > 0 required"))

    Φ⁻¹(x::T) = T(norminvcdf(Float64(x)))
    ϕ(x::T) = T(normpdf(Float64(x)))
    Φ(x::T) = T(normcdf(Float64(x)))

    σ_z = σ / sqrt(one(T) - ρ * ρ)
    invσ_z = one(T) / σ_z

    q = range(zero(T), one(T); length=N + 1)
    m = μ .+ σ_z .* (Φ⁻¹).(q)

    z = Vector{T}(undef, N)
    @inbounds for i in 1:N
        lo = (m[i] - μ) * invσ_z
        hi = (m[i+1] - μ) * invσ_z
        z[i] = μ - σ_z * T(N) * (ϕ(hi) - ϕ(lo))
    end

    if isodd(N)
        z[(N+1)÷2] = μ
    end

    if iszero(ρ)
        π = fill(inv(T(N)), N)
        return z, π
    end

    dens_z = (zv::T) -> ϕ((zv - μ) * invσ_z) * invσ_z
    drift = μ * (one(T) - ρ)

    Π = Matrix{T}(undef, N, N)
    @inbounds for i in 1:N, j in 1:N
        integrand = function (zv::T)
            lo = (m[j] - drift - ρ * zv) / σ
            hi = (m[j+1] - drift - ρ * zv) / σ
            pj = Φ(hi) - Φ(lo)
            return dens_z(zv) * pj
        end
        val = quadgk(integrand, m[i], m[i+1]; rtol=rtol, atol=atol)[1]
        Π[i, j] = T(N) * val
    end

    @inbounds @views for i in 1:N
        s = sum(Π[i, :])
        Π[i, :] ./= s
    end

    return z, Π
end

function initialize_static_parameters(;
    e1_size::Int64=3,           # number of permanent shock states
    e1_σ::Float64=0.448,        # std. dev. of permanent shock
    e2_size::Int64=3,           # number of persistent shock states
    e2_ρ::Float64=0.957,        # persistence of AR(1) shock
    e2_σ::Float64=0.129,        # std. dev. of AR(1) innovation
    e3_size::Int64=3,           # number of transitory shock states
    e3_σ::Float64=0.351,        # std. dev. of transitory i.i.d. shock
    a_max::Float64=800.0,       # max asset on positive grid
    a_size_neg_1::Int64=51,     # count of (a'≤-1) asset grid points for VFI
    a_size_neg_2::Int64=101,    # count of (-1≤a'≤0) asset grid points for VFI
    a_size_pos_1::Int64=101,    # count of (1≥a'≥0) asset grid points for VFI
    a_size_pos_2::Int64=51,     # count of (a'≥1) asset grid points for VFI
    a_degree_neg::Int64=2,      # curvature exponent for negative grid
    a_degree_pos::Int64=2       # curvature exponent for positive grid
)

    e1_grid, e1_G = adda_cooper(e1_size, 0.0, e1_σ)
    e1_Γ = Matrix{Float64}(I, e1_size, e1_size)
    exp_e1_grid = exp.(e1_grid)
    # exp_e1_grid = exp_e1_grid ./ sum(exp_e1_grid .* e1_G)

    inv_e2_σ = 1.0 / e2_σ
    e2_MC = rouwenhorst(e2_size, e2_ρ, e2_σ, 0.0)
    # e2_MC = tauchen(e2_size, e2_ρ, e2_σ, 0.0, 4.0)
    e2_Γ = e2_MC.p
    e2_G = stationary_distributions(e2_MC)[1]
    e2_grid = collect(e2_MC.state_values)
    # e2_grid, e2_Γ = adda_cooper(e2_size, e2_ρ, e2_σ)
    # e2_G = stationary_distributions(MarkovChain(e2_Γ, e2_grid))[1]
    exp_e2_grid = exp.(e2_grid)
    # exp_e2_grid = exp_e2_grid ./ sum(exp_e2_grid .* e2_G)

    e3_grid, e3_G = adda_cooper(e3_size, 0.0, e3_σ)
    e3_Γ = e3_G
    exp_e3_grid = exp.(e3_grid)
    # exp_e3_grid = exp_e3_grid ./ sum(exp_e3_grid .* e3_G)

    e13_grid = [e1 + e3 for e3 in e3_grid, e1 in e1_grid]
    exp_e13_grid = exp.(e13_grid)
    # exp_e13_grid = [exp_e1 * exp_e3 for exp_e3 in exp_e3_grid, exp_e1 in exp_e1_grid]

    e123_grid = [e1 + e2 + e3 for e3 in e3_grid, e2 in e2_grid, e1 in e1_grid]
    exp_e123_grid = exp.(e123_grid)
    # exp_e123_grid = [exp_e1 * exp_e2 * exp_e3 for exp_e3 in exp_e3_grid, exp_e2 in exp_e2_grid, exp_e1 in exp_e1_grid]

    E = sum(exp_e123_grid .*
            reshape(e1_G, (1, 1, e1_size)) .*
            reshape(e2_G, (1, e2_size, 1)) .*
            reshape(e3_G, (e3_size, 1, 1)))

    a_min = -1.0 * exp_e1_grid[end] * exp_e2_grid[end] * exp_e3_grid[end]

    a_grid_neg_1 = ((range(start=a_size_neg_1 - 1, stop=0.0, length=a_size_neg_1) ./ (a_size_neg_1 - 1)) .^ a_degree_neg) .* (a_min + 1.0) .- 1.0
    a_grid_neg_2 = collect(range(start=-1.0, stop=0.0, length=a_size_neg_2))
    a_grid_neg = vcat(a_grid_neg_1[1:(end-1)], a_grid_neg_2[1:(end-1)])
    a_size_neg = length(a_grid_neg)

    a_grid_pos_1 = collect(range(start=0.0, stop=1.0, length=a_size_pos_1))
    a_grid_pos_2 = ((range(start=0.0, stop=a_size_pos_2 - 1, length=a_size_pos_2) ./ (a_size_pos_2 - 1)) .^ a_degree_pos) .* (a_max - 1.0) .+ 1.0
    a_grid_pos = vcat(a_grid_pos_1[1:(end-1)], a_grid_pos_2)
    a_size_pos = length(a_grid_pos)

    a_grid = vcat(a_grid_neg, a_grid_pos)
    a_size = length(a_grid)
    a_ind_zero = a_size_neg + 1

    e2_μ_grid = e2_ρ .* e2_grid
    e2_μ_σ2_grid = e2_μ_grid .+ 0.5 * e2_σ^2
    exp_e2_μ_σ2_grid = exp.(e2_μ_σ2_grid)

    loop_e2_e1 = CartesianIndices((e2_size, e1_size))
    loop_e3_e2_e1 = CartesianIndices((e3_size, e2_size, e1_size))

    loop_a_e2_e1 = CartesianIndices((a_size, e2_size, e1_size))
    loop_a_neg_e2_e1 = CartesianIndices((a_size_neg, e2_size, e1_size))
    loop_a_neg_e3_e1 = CartesianIndices((a_size_neg, e3_size, e1_size))
    loop_a_pos_e2_e1 = CartesianIndices((a_size_pos, e2_size, e1_size))

    return (
        e1_size=e1_size,
        e1_σ=e1_σ,
        e1_G=e1_G,
        e1_Γ=e1_Γ,
        e1_grid=e1_grid,
        exp_e1_grid=exp_e1_grid,
        e2_size=e2_size,
        e2_ρ=e2_ρ,
        e2_σ=e2_σ,
        inv_e2_σ=inv_e2_σ,
        e2_G=e2_G,
        e2_Γ=e2_Γ,
        e2_grid=e2_grid,
        exp_e2_grid=exp_e2_grid,
        e3_size=e3_size,
        e3_σ=e3_σ,
        e3_G=e3_G,
        e3_Γ=e3_Γ,
        e3_grid=e3_grid,
        exp_e3_grid=exp_e3_grid,
        e13_grid=e13_grid,
        exp_e13_grid=exp_e13_grid,
        e123_grid=e123_grid,
        exp_e123_grid=exp_e123_grid,
        E=E,
        a_min=a_min,
        a_max=a_max,
        a_grid=a_grid,
        a_grid_neg=a_grid_neg,
        a_grid_pos=a_grid_pos,
        a_size=a_size,
        a_size_neg=a_size_neg,
        a_size_pos=a_size_pos,
        a_ind_zero=a_ind_zero,
        a_degree_neg=a_degree_neg,
        a_degree_pos=a_degree_pos,
        e2_μ_grid=e2_μ_grid,
        e2_μ_σ2_grid=e2_μ_σ2_grid,
        exp_e2_μ_σ2_grid=exp_e2_μ_σ2_grid,
        loop_e2_e1=loop_e2_e1,
        loop_e3_e2_e1=loop_e3_e2_e1,
        loop_a_e2_e1=loop_a_e2_e1,
        loop_a_neg_e2_e1=loop_a_neg_e2_e1,
        loop_a_neg_e3_e1=loop_a_neg_e3_e1,
        loop_a_pos_e2_e1=loop_a_pos_e2_e1,
    )
end

function initialize_tuned_parameters(static_parameters::NamedTuple;
    ρ::Float64=0.975,                   # survival rate (40 years)
    r_f::Float64=0.04,                  # risk-free rate
    β::Float64=0.92,                    # discount factor (households) # 1.0 / (ρ * (1.0 + r_f))
    β_f::Float64=β,                     # discount factor (bank)
    τ::Float64=0.04,                    # transaction cost
    γ::Float64=3.00,                    # CRRA coefficient
    δ::Float64=0.10,                    # depreciation rate
    α::Float64=0.36,                    # capital share
    ψ::Float64=0.972^4,                 # exogenous retention ratio # 1.0 - 1.0 / 20.0
    θ::Float64=1.0 / (4.57 * 0.75),     # diverting fraction # 1.0 / 3.0
    Ph::Float64=1.0 / 6.0,              # prob. of history erased
    η::Float64=0.35,                    # wage garnishment rate
    ζ::Float64=0.001,                   # EV shock scale
    κ::Float64=697 / 33176,             # out-of-pocket monetary filing cost
    λ::Float64=0.0                      # multiplier
)

    @unpack e3_size, e3_Γ, e2_size, e2_Γ, e1_size, exp_e123_grid, E = static_parameters
    @unpack a_size, a_grid, a_grid_neg = static_parameters

    inv_ζ = 1.0 / ζ

    ξ_λ = (1.0 - ψ) / (1.0 - λ - ψ)
    Λ_λ = β_f * (1.0 - ψ + ψ * ξ_λ)
    LR_λ = ξ_λ / θ
    AD_λ = LR_λ / (LR_λ - 1.0)
    ι_λ = λ * θ / Λ_λ
    r_k_λ = r_f + ι_λ
    # E = 1.0
    K_λ = E * ((r_k_λ + δ) / α)^(1.0 / (α - 1.0))
    w_λ = (1.0 - α) * (K_λ / E)^α

    q_bar = ρ / (1.0 + r_f)
    R_bar = ρ ./ ((-a_grid_neg) .* ((1.0 + r_f) * (1.0 + τ) + ι_λ))
    Γ_default = zeros(e3_size, e2_size, e1_size)
    for e1_i = 1:e1_size, e2_i = 1:e2_size, e3_i = 1:e3_size
        exp_e123 = exp_e123_grid[e3_i, e2_i, e1_i]
        Γ_default[e3_i, e2_i, e1_i] = η * w_λ * exp_e123
    end

    ρβ = ρ * β
    Γ = zeros(e3_size, e2_size, e2_size)
    for e2_i in 1:e2_size, e2_p_i in 1:e2_size, e3_p_i in 1:e3_size
        Γ[e3_p_i, e2_p_i, e2_i] = e3_Γ[e3_p_i] * e2_Γ[e2_i, e2_p_i]
    end
    Γ_ρβ = ρβ .* Γ
    bellman_factor = 1.0 - ρβ

    W = zeros(e3_size, e2_size, e1_size)
    WA = zeros(a_size, e3_size, e2_size, e1_size)
    c_d = zeros(e3_size, e2_size, e1_size)
    u_d = zeros(e3_size, e2_size, e1_size)
    for e1_i = 1:e1_size, e2_i = 1:e2_size, e3_i = 1:e3_size
        exp_e123 = exp_e123_grid[e3_i, e2_i, e1_i]
        W_temp = w_λ * exp_e123
        W[e3_i, e2_i, e1_i] = W_temp
        WA[:, e3_i, e2_i, e1_i] .= W_temp .+ a_grid
        c_d_temp = (1.0 - η) * W_temp - κ
        c_d[e3_i, e2_i, e1_i] = c_d_temp
        u_d[e3_i, e2_i, e1_i] = utility(c_d_temp, γ, bellman_factor)
    end

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
        ζ=ζ,
        inv_ζ=inv_ζ,
        κ=κ,
        λ=λ,
        ξ_λ=ξ_λ,
        Λ_λ=Λ_λ,
        LR_λ=LR_λ,
        AD_λ=AD_λ,
        ι_λ=ι_λ,
        r_k_λ=r_k_λ,
        K_λ=K_λ,
        w_λ=w_λ,
        q_bar=q_bar,
        R_bar=R_bar,
        Γ_default=Γ_default,
        ρβ=ρβ,
        Γ=Γ,
        Γ_ρβ=Γ_ρβ,
        bellman_factor=bellman_factor,
        W=W,
        WA=WA,
        c_d=c_d,
        u_d=u_d,
    )
end

@inline function utility(c::Float64, γ::Float64, bellman_factor::Float64)

    if c > 0.0
        return γ == 1.0 ? bellman_factor * log(c) : bellman_factor * 1.0 / ((1.0 - γ) * c^(γ - 1.0))
    else
        return -1E+12
    end
end

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
            best_f, best_a, best_i = f0, a0, i
        end

        f1 = a1 * q1
        if f1 < best_f
            best_f, best_a, best_i = f1, a1, i
        end
    end

    return best_a, best_f, best_i
end

mutable struct MutableAggregateVariables{T}
    K::T
    L::T
    A::T
    D::T
    N::T
    LR::T
    AD::T
    profit::T
    ω::T
    share_of_filers::T
    share_in_debts::T
    debt_to_earning_ratio::T
    avg_loan_rate::T
end

mutable struct MutableVariables{T,
    A2<:AbstractArray{T,2},
    A3<:AbstractArray{T,3},
    A4<:AbstractArray{T,4}}
    aggregate_variables::MutableAggregateVariables{T}
    R::A3
    q::A3
    rbl_a::A2
    rbl_qa::A2
    V::A4
    V_d::A3
    V_nd::A4
    V_pos::A4
    EV::A3
    EV_pos::A3
    EV_Ph::A3
    policy_a::A4
    policy_d::A4
    policy_a_pos::A4
end

@inline @views @inbounds function repayment_mat(thres_e2::Float64, a_p_i::Int64, e2_i::Int64, e1_i::Int64, parameters::NamedTuple)

    @unpack e2_μ_grid, inv_e2_σ, e2_σ, a_grid_neg, Γ_default = parameters

    e2_μ = e2_μ_grid[e2_i]
    a_p = a_grid_neg[a_p_i]
    z_e2 = (thres_e2 - e2_μ) * inv_e2_σ
    repay_amount = (-a_p) * normcdf(-z_e2)
    default_amount = Γ_default[end, e2_i, e1_i] * normcdf(z_e2 - e2_σ)
    total_amount = repay_amount + default_amount
    return clamp(total_amount, 0.0, -a_p)
end

@inline log_(thres_e::Float64) = thres_e > 0.0 ? log(thres_e) : -Inf

@views @inbounds function create_variables(parameters::NamedTuple; T::Type{<:Real}=Float64)

    @unpack a_size, a_size_neg, a_size_pos, a_grid_neg = parameters
    @unpack e1_size, e1_grid, e2_size, e3_size = parameters
    @unpack q_bar, R_bar, κ, η, w_λ, loop_a_neg_e2_e1, loop_e2_e1 = parameters

    aggregate_variables = MutableAggregateVariables{T}(
        zero(T), zero(T), zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T))

    R = Array{T}(undef, a_size_neg, e2_size, e1_size)
    q = fill(T(q_bar), a_size, e2_size, e1_size)
    @batch for idx in loop_a_neg_e2_e1
        a_p_i, e2_i, e1_i = idx.I
        e1 = e1_grid[e1_i]
        a_neg = a_grid_neg[a_p_i]
        thres_e2_ = log_((-a_neg - κ) / (η * w_λ)) - e1
        R_temp = repayment_mat(thres_e2_, a_p_i, e2_i, e1_i, parameters)
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

    V = zeros(T, a_size, e3_size, e2_size, e1_size)
    V_d = Array{T}(undef, e3_size, e2_size, e1_size)
    V_nd = Array{T}(undef, a_size, e3_size, e2_size, e1_size)
    V_pos = zeros(T, a_size_pos, e3_size, e2_size, e1_size)

    EV = zeros(T, a_size, e2_size, e1_size)
    EV_pos = zeros(T, a_size_pos, e2_size, e1_size)
    EV_Ph = zeros(T, a_size_pos, e2_size, e1_size)

    policy_a = Array{T}(undef, a_size, e3_size, e2_size, e1_size)
    policy_d = Array{T}(undef, a_size, e3_size, e2_size, e1_size)
    policy_a_pos = Array{T}(undef, a_size_pos, e3_size, e2_size, e1_size)

    return MutableVariables{T,
        typeof(rbl_a),typeof(R),typeof(V)}(
        aggregate_variables,
        R, q, rbl_a, rbl_qa,
        V, V_d, V_nd, V_pos, EV, EV_pos, EV_Ph,
        policy_a, policy_d, policy_a_pos,
    )
end

struct ItpCache{ItpQ,ItpEv,ItpEvPh}
    q::Array{ItpQ,2}            # size: (e2_size, e1_size)
    EV::Array{ItpEv,2}          # size: (e2_size, e1_size)
    EV_Ph::Array{ItpEvPh,2}     # size: (e2_size, e1_size)
end

@inline build_itp(xs, ys) = linear_interpolation(xs, ys, extrapolation_bc=Interpolations.Flat())

@views @inbounds @views function build_itp_cache(variables::MutableVariables, parameters::NamedTuple)

    @unpack a_grid, a_grid_pos, e1_size, e2_size = parameters

    q_ = variables.q[:, 1, 1]
    q_sample = build_itp(a_grid, q_)
    EV_ = variables.EV[:, 1, 1]
    EV_sample = build_itp(a_grid, EV_)
    EV_Ph_ = variables.EV_Ph[:, 1, 1]
    EV_Ph_sample = build_itp(a_grid_pos, EV_Ph_)

    q_itp = Array{typeof(q_sample)}(undef, e2_size, e1_size)
    EV_itp = Array{typeof(EV_sample)}(undef, e2_size, e1_size)
    EV_Ph_itp = Array{typeof(EV_Ph_sample)}(undef, e2_size, e1_size)

    for e2_i in 1:e2_size, e1_i in 1:e1_size
        q_ = variables.q[:, e2_i, e1_i]
        q_itp[e2_i, e1_i] = build_itp(a_grid, q_)
        EV_ = variables.EV[:, e2_i, e1_i]
        EV_itp[e2_i, e1_i] = build_itp(a_grid, EV_)
        EV_Ph_ = variables.EV_Ph[:, e2_i, e1_i]
        EV_Ph_itp[e2_i, e1_i] = build_itp(a_grid_pos, EV_Ph_)
    end

    return ItpCache{typeof(q_sample),typeof(EV_sample),typeof(EV_Ph_sample)}(q_itp, EV_itp, EV_Ph_itp)
end

function update_EV!(V_p::Array{Float64,4}, V_pos_p::Array{Float64,4}, variables::MutableVariables, parameters::NamedTuple)

    @unpack a_ind_zero, Ph, Γ_ρβ, loop_a_e2_e1, loop_a_pos_e2_e1 = parameters

    @views @inbounds @batch for idx in loop_a_e2_e1
        a_p_i, e2_i, e1_i = idx.I
        Γ_ρβ_temp = Γ_ρβ[:, :, e2_i]
        V_p_temp = V_p[a_p_i, :, :, e1_i]
        variables.EV[a_p_i, e2_i, e1_i] = sum(Γ_ρβ_temp .* V_p_temp)
    end

    @views @inbounds @batch for idx in loop_a_pos_e2_e1
        a_pos_p_i, e2_i, e1_i = idx.I
        a_p_i = a_pos_p_i + a_ind_zero - 1
        EV_temp = variables.EV[a_p_i, e2_i, e1_i]
        Γ_ρβ_temp = Γ_ρβ[:, :, e2_i]
        V_pos_p_temp = V_pos_p[a_pos_p_i, :, :, e1_i]
        EV_pos_temp = sum(Γ_ρβ_temp .* V_pos_p_temp)
        variables.EV_pos[a_pos_p_i, e2_i, e1_i] = EV_pos_temp
        variables.EV_Ph[a_pos_p_i, e2_i, e1_i] = Ph * EV_temp + (1.0 - Ph) * EV_pos_temp
    end

    return nothing
end

function update_V_d!(variables::MutableVariables, parameters::NamedTuple)

    @unpack loop_e3_e2_e1, u_d = parameters

    @inbounds @batch for idx in loop_e3_e2_e1
        e3_i, e2_i, e1_i = idx.I
        EV_pos_zero = variables.EV_pos[1, e2_i, e1_i]
        u_d_temp = u_d[e3_i, e2_i, e1_i]
        variables.V_d[e3_i, e2_i, e1_i] = u_d_temp + EV_pos_zero
    end

    return nothing
end

struct DP_Problem{ItpQ,ItpEv,FunU,T}
    qa_itp::ItpQ
    EV_itp::ItpEv
    utility::FunU
    γ::T
    bellman_factor::T
end

@inline function obj_DP(DP::DP_Problem, a_p::Float64, a::Float64, W_::Float64)
    c = W_ + a - DP.qa_itp(a_p)
    return -(DP.utility(c, DP.γ, DP.bellman_factor) + DP.EV_itp(a_p))
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
    res = Optim.optimize(F, lb, ub, Optim.GoldenSection(); rel_tol=rtol, abs_tol=atol, iterations=iters)
    # res = Optim.optimize(F, lb, ub, Optim.Brent(); rel_tol=rtol, abs_tol=atol, iterations=iters)

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

function update_value_and_policy_functions!(
    V_p::Array{Float64,4},
    V_pos_p::Array{Float64,4},
    variables::MutableVariables,
    parameters::NamedTuple,
    itp_cache::ItpCache
)
    """
    one-step update of value and policy functions
    """

    @unpack a_size, a_grid, a_size_pos, a_grid_pos, a_ind_zero, a_min, a_max = parameters
    @unpack e1_size, e1_grid, e1_Γ, e2_size, e2_grid, e2_Γ, e3_size, e3_grid, e3_Γ = parameters
    @unpack ρ, β, γ, r_f, Ph, η, κ, W, q_bar, ζ, inv_ζ, bellman_factor = parameters
    @unpack c_d, loop_e2_e1 = parameters

    update_EV!(V_p, V_pos_p, variables, parameters)
    update_V_d!(variables, parameters)

    @views @inbounds @batch for idx in loop_e2_e1

        e2_i, e1_i = idx.I

        q_ = variables.q[:, e2_i, e1_i]
        q_itp = itp_cache.q[e2_i, e1_i]
        copyto!(q_itp.itp.coefs, q_)
        qa_itp = QaInterpolant(q_itp)

        rbl_a_ = variables.rbl_a[e2_i, e1_i]
        rbl_qa_ = variables.rbl_qa[e2_i, e1_i]

        EV_ = variables.EV[:, e2_i, e1_i]
        EV_itp = itp_cache.EV[e2_i, e1_i]
        copyto!(EV_itp.itp.coefs, EV_)

        EV_Ph_ = variables.EV_Ph[:, e2_i, e1_i]
        EV_Ph_itp = itp_cache.EV_Ph[e2_i, e1_i]
        copyto!(EV_Ph_itp.itp.coefs, EV_Ph_)

        DP_Problem_nd = DP_Problem(qa_itp, EV_itp, utility, γ, bellman_factor)
        DP_Problem_pos = DP_Problem(qa_itp, EV_Ph_itp, utility, γ, bellman_factor)

        for e3_i = 1:e3_size

            W_ = W[e3_i, e2_i, e1_i]
            V_d_ = variables.V_d[e3_i, e2_i, e1_i]

            lb_nd = max(1.05 * rbl_a_, a_min)
            lb_pos = 0.0

            for a_i = 1:a_size

                a = a_grid[a_i]
                CoH = W_ + a
                ub_q = min(CoH / q_bar, a_max)
                budget_gap = CoH - rbl_qa_
                bound_gap = ub_q - lb_nd

                if (budget_gap ≤ 0.0) || (bound_gap ≤ 0.0)
                    variables.V_nd[a_i, e3_i, e2_i, e1_i] = -1E+12
                    variables.V[a_i, e3_i, e2_i, e1_i] = V_d_
                    variables.policy_d[a_i, e3_i, e2_i, e1_i] = 1.0
                    variables.policy_a[a_i, e3_i, e2_i, e1_i] = 0.0
                else
                    V_nd_, a_star_nd, _ = solve_DP(DP_Problem_nd, a, W_; lb=lb_nd, ub=ub_q)
                    variables.V_nd[a_i, e3_i, e2_i, e1_i] = V_nd_
                    variables.policy_a[a_i, e3_i, e2_i, e1_i] = a_star_nd
                    if a ≥ 0.0
                        variables.V[a_i, e3_i, e2_i, e1_i] = V_nd_
                        variables.policy_d[a_i, e3_i, e2_i, e1_i] = 0.0
                    else
                        Δ = (V_d_ - V_nd_) * inv_ζ
                        x = -abs(Δ)
                        m = max(V_d_, V_nd_)
                        V_ = m + ζ * log1p(exp(x))
                        policy_d_ = 0.5 * (1 + tanh(0.5 * Δ))
                        variables.V[a_i, e3_i, e2_i, e1_i] = V_
                        variables.policy_d[a_i, e3_i, e2_i, e1_i] = policy_d_
                    end
                end

                if a_i >= a_ind_zero
                    a_pos_i = a_i - a_ind_zero + 1
                    V_pos_, a_star_pos, _ = solve_DP(DP_Problem_pos, a, W_; lb=lb_pos, ub=ub_q)
                    variables.V_pos[a_pos_i, e3_i, e2_i, e1_i] = V_pos_
                    variables.policy_a_pos[a_pos_i, e3_i, e2_i, e1_i] = a_star_pos
                end
            end
        end
    end

    return nothing
end

@inline @views @inbounds function repayment_mat(policy_d_::AbstractArray{Float64,2}, a_p_i::Int64, e1_i::Int64, parameters::NamedTuple)::Matrix{Float64}

    @unpack e2_μ_grid, inv_e2_σ, e2_σ, a_grid_neg, Γ_default = parameters

    a_p = a_grid_neg[a_p_i]
    Γ_default_ = Γ_default[:, :, e1_i]
    repay_amount = (-a_p) .* (1.0 .- policy_d_)
    default_amount = Γ_default_ .* policy_d_
    total_amount = repay_amount + default_amount
    return clamp.(total_amount, 0.0, -a_p)
end

function update_pricing_and_rbl_functions!(variables::MutableVariables, parameters::NamedTuple)

    @unpack loop_a_neg_e2_e1, Γ, R_bar, loop_e2_e1, a_size_neg, a_grid_neg = parameters

    @views @inbounds @batch for idx in loop_a_neg_e2_e1
        a_p_i, e2_i, e1_i = idx.I
        policy_d_ = variables.policy_d[a_p_i, :, :, e1_i]
        Γ_ = Γ[:, :, e2_i]
        repayment_ = repayment_mat(policy_d_, a_p_i, e1_i, parameters)
        R_temp = sum(Γ_ .* repayment_)
        variables.R[a_p_i, e2_i, e1_i] = R_temp
        variables.q[a_p_i, e2_i, e1_i] = R_bar[a_p_i] * R_temp
    end

    @views @inbounds @batch for idx in loop_e2_e1
        e2_i, e1_i = idx.I
        q_grid_neg = variables.q[1:a_size_neg, e2_i, e1_i]
        rbl_a_, rbl_qa_, _ = find_min_qa(a_grid_neg, q_grid_neg)
        variables.rbl_a[e2_i, e1_i] = rbl_a_
        variables.rbl_qa[e2_i, e1_i] = rbl_qa_
    end

    return nothing
end

# safe_abs(x) = ifelse(isnan(x), 0.0, abs(x))

function solve_value_and_policy_functions!(variables::MutableVariables, itp_cache::ItpCache, parameters::NamedTuple;
    tol::Float64=1E-6, iter_max::Int64=500, relax_V::Float64=1.0, relax_q::Float64=1.0, bellman_step::Int64=1)

    @assert 0.0 < relax_V ≤ 1.0 "relaxation must be in (0,1]; got $relax_V"
    @assert 0.0 < relax_q ≤ 1.0 "relaxation must be in (0,1]; got $relax_q"
    @assert bellman_step ≥ 1 "bellman step has to be larger than or equal to one; got $bellman_step"

    r0V, r1V = 1.0 - relax_V, relax_V
    r0q, r1q = 1.0 - relax_q, relax_q
    search_iter = 0
    V_crit = Inf
    V_pos_crit = Inf
    q_crit = Inf
    crit = Inf
    prog = ProgressThresh(tol, "Solving value and policy functions (one-loop): ")

    V_p = similar(variables.V)
    # V_nd_p = similar(variables.V_nd)
    # V_d_p = similar(variables.V_d)
    V_pos_p = similar(variables.V_pos)
    q_p = similar(variables.q)

    while crit > tol && search_iter < iter_max

        copyto!(V_p, variables.V)
        # copyto!(V_nd_p, variables.V_nd)
        # copyto!(V_d_p, variables.V_d)
        copyto!(V_pos_p, variables.V_pos)
        copyto!(q_p, variables.q)

        bellman_step_ = q_crit ≤ 1E-4 ? bellman_step : 1
        for _ in 1:bellman_step_
            update_value_and_policy_functions!(V_p, V_pos_p, variables, parameters, itp_cache)
            @. variables.V = r0V * V_p + r1V * variables.V
            @. variables.V_pos = r0V * V_pos_p + r1V * variables.V_pos
        end

        update_value_and_policy_functions!(V_p, V_pos_p, variables, parameters, itp_cache)
        update_pricing_and_rbl_functions!(variables, parameters)

        # V_crit = maximum(abs, @. variables.V - V_p)
        V_crit, V_crit_i = findmax(@. abs(variables.V - V_p))
        # V_nd_crit = maximum(safe_abs, @. variables.V_nd - V_nd_p)
        # V_d_crit = maximum(safe_abs, @. variables.V_d - V_d_p)
        V_pos_crit = maximum(abs, @. variables.V_pos - V_pos_p)
        q_crit = maximum(abs, @. variables.q - q_p)
        crit = max(V_crit, V_pos_crit, q_crit)
        # crit = max(V_nd_crit, V_d_crit, V_pos_crit, q_crit)
        # crit = q_crit

        ProgressMeter.update!(prog, crit)
        search_iter += 1

        # println("$V_crit at $V_crit_i")

        @. variables.q = r0q * q_p + r1q * variables.q
    end

    println("$V_crit, $V_pos_crit, $q_crit")

    return crit
end

static_parameters = initialize_static_parameters();
tuned_parameters = initialize_tuned_parameters(static_parameters; λ = 0.0); # kwargs = (β = 0.99, λ = 0.01) ; kwargs...
parameters = (; static_parameters..., tuned_parameters...);
variables = create_variables(parameters);
itp_cache = build_itp_cache(variables, parameters);
solve_value_and_policy_functions!(variables, itp_cache, parameters; tol=1E-6, relax_V=1.0, relax_q=1.0, bellman_step=1);

e1_ind = 3
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, e1_ind])