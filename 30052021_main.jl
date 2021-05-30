#===========================#
# Import packages and files #
#===========================#
using Distributions
using LinearAlgebra: norm
using Parameters: @unpack
using Plots
using ProgressMeter
using Optim
using QuantEcon: rouwenhorst, tauchen, stationary_distributions

# load user-defined files
include("interpolate.jl")
include("smooth.jl")

# print out the number of threads
println("Julia is running with $(Threads.nthreads()) threads...")

#==================#
# Define functions #
#==================#
function parameters_function(;
    β::Real = 0.96,             # discount factor (households)
    β_f::Real = 0.96,           # discount factor (bank)
    r_f::Real = 1.0/β_f-1.0,   # risk-free rate
    σ::Real = 2.00,             # CRRA coefficient
    η::Real = 0.40,             # garnishment rate
    δ::Real = 0.08,             # depreciation rate
    α::Real = 1.0/3.0,          # capital share
    ψ::Real = 0.90,             # exogenous dividend rate
    λ::Real = 0.00,             # multiplier of incentive constraint
    θ::Real = 0.40,             # diverting fraction
    e_ρ::Real = 0.95,           # AR(1) of endowment shock
    e_σ::Real = 0.10,           # s.d. of endowment shock
    e_size::Integer = 9,        # number of endowment shock
    ν_s::Real = 0.95,           # scale of patience
    ν_p::Real = 0.10,           # probability of patience
    ν_size::Integer = 2,        # number of preference shock
    a_min::Real = -5.0,         # min of asset holding
    a_max::Real = 350.0,        # max of asset holding
    a_size_neg::Integer = 501,  # number of grid of negative asset holding for VFI
    a_size_pos::Integer = 351,  # number of grid of positive asset holding for VFI
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
    ν_Γ = [ν_p, 1.0-ν_p]

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length = a_size_neg))
    a_grid_pos = ((range(0.0, stop = a_size_pos-1, length = a_size_pos)/(a_size_pos-1)).^a_degree)*a_max
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)
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
    ξ = (1.0-ψ)/(1-λ-ψ)
    Λ = β_f*(1.0-ψ+ψ*ξ)
    LR = ξ/θ
    AD = LR/(LR-1.0)
    ι = λ*θ/Λ
    r_k = r_f + ι
    E = exp(e_SS)
    K = E*((r_k+δ)/α)^(1.0/(α-1.0))
    w = (1.0-α)*(K/E)^α

    # return values
    return (β = β, β_f = β_f, r_f = r_f, σ = σ, η = η, δ = δ, α = α, ψ = ψ,
            λ = λ, θ = θ,
            a_degree = a_degree, μ_scale = μ_scale,
            e_ρ = e_ρ, e_σ = e_σ, e_size = e_size, e_Γ = e_Γ, e_grid = e_grid,
            ν_s = ν_s, ν_p = ν_p, ν_size = ν_size, ν_Γ = ν_Γ, ν_grid = ν_grid,
            a_grid = a_grid, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos,
            a_size = a_size, a_size_neg = a_size_neg, a_size_pos = a_size_pos,
            a_ind_zero = a_ind_zero,
            a_grid_μ = a_grid_μ, a_grid_neg_μ = a_grid_neg_μ, a_grid_pos_μ = a_grid_pos_μ,
            a_size_μ = a_size_μ, a_size_neg_μ = a_size_neg_μ, a_size_pos_μ = a_size_pos_μ,
            a_ind_zero_μ = a_ind_zero_μ,
            ξ = ξ, Λ = Λ, LR = LR, AD = AD, ι = ι, r_k = r_k, E = E, K = K, w = w)
end

mutable struct MutableAggregateVariables
    """
    construct a type for mutable variables
    """
    K::Real
    L::Real
    N::Real
end

mutable struct MutableVariables
    """
    construct a type for mutable functions
    """
    V::Array{Float64,3}
    V_d::Array{Float64,2}
    V_nd::Array{Float64,3}
    policy_a::Array{Float64,3}
    policy_d::Array{Float64,3}
    p::Array{Float64,2}
    q::Array{Float64,2}
    μ::Array{Float64,3}
    aggregate_variables::MutableAggregateVariables
end

function utility_function(
    c::Real,
    γ::Real
    )
    """
    compute utility of CRRA utility function with coefficient γ
    """

    if c > 0.0
        return γ == 1.0 ? log(c) : 1.0 / ((1.0-γ)*c^(γ-1.0))
    else
        return -Inf
    end
end

function solve_ED_function(
    parameters::NamedTuple;
    tol::Real = tol,
    iter_max::Integer = iter_max
    )
    """
    solve the economy where enforced defualt (ED) is imposed based on income
    """

    # unpack parameters
    @unpack a_size, a_size_neg, a_grid, a_grid_neg, a_ind_zero = parameters
    @unpack ν_size, ν_grid, ν_Γ = parameters
    @unpack e_size, e_grid, e_Γ, e_ρ, e_σ = parameters
    @unpack β, σ, r_f, ι, η, w = parameters

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving ED economy for initialization: ")

    # compute equilibrium repament probability and pricing functions
    p = ones(a_size, e_size)
    q = ones(a_size, e_size)
    for e_i in 1:e_size
        @inbounds e_μ = e_ρ*e_grid[e_i]
        @inbounds @views p[:,e_i] = 1.0 .- cdf.(LogNormal(e_μ,e_σ),-a_grid)
        @inbounds @views q[:,e_i] = p[:,e_i]./(1.0+r_f+ι)
    end
    q_function(a_p,e_μ,e_σ) = (1.0-cdf(LogNormal(e_μ,e_σ),-a_p))/(1.0+r_f+ι)
    qa_function(a_p,e_μ,e_σ) = q_function(a_p,e_μ,e_σ)*a_p

    # initialize containers
    V = zeros(a_size, ν_size, e_size)
    V_p = similar(V)
    V_d = zeros(ν_size, e_size)
    V_nd = zeros(a_size, ν_size, e_size)

    # solve eqquilibrium value functions
    while crit > tol && iter < iter_max
        # copy the current value function to the pre-specified container
        copyto!(V_p, V)

        # update household's problem
        Threads.@threads for e_i in 1:e_size

            # extract endowment
            @inbounds e = e_grid[e_i]

            # compute the next-period discounted expected value funtions and interpolated functions
            @inbounds @views V_expt_p = (ν_Γ[1]*V_p[:,1,:] + ν_Γ[2]*V_p[:,2,:])*e_Γ[e_i,:]
            @inbounds @views V_hat_impatient = ν_grid[1]*β*V_expt_p
            @inbounds @views V_hat_patient = ν_grid[2]*β*V_expt_p
            V_hat_impatient_itp = Akima(a_grid, V_hat_impatient)
            V_hat_patient_itp = Akima(a_grid, V_hat_patient)

            # compute defaulting value
            @inbounds V_d[1,e_i] = utility_function((1-η)*w*exp(e),σ) + V_hat_impatient[a_ind_zero]
            @inbounds V_d[2,e_i] = utility_function((1-η)*w*exp(e),σ) + V_hat_patient[a_ind_zero]

            # find risky borrowing limit
            object_rbl(a_p) = qa_function(a_p,e_ρ*e,e_σ)
            res_rbl = optimize(object_rbl, a_grid[1], 0.0)
            rbl = Optim.minimizer(res_rbl)

            # compute non-defaulting value
            Threads.@threads for a_i in 1:a_size
                @inbounds CoH = w*exp(e) + a_grid[a_i]
                if (CoH - object_rbl(rbl)) >= 0.0
                    object_nd_impatient(a_p) = -(utility_function(CoH-object_rbl(a_p),σ) + V_hat_impatient_itp(a_p))
                    res_nd_impatient = optimize(object_nd_impatient, rbl, CoH*(1+r_f+ι))
                    @inbounds V_nd[a_i,1,e_i] = -Optim.minimum(res_nd_impatient)
                    object_nd_patient(a_p) = -(utility_function(CoH-object_rbl(a_p),σ) + V_hat_patient_itp(a_p))
                    res_nd_patient = optimize(object_nd_patient, rbl, CoH*(1+r_f+ι))
                    @inbounds V_nd[a_i,2,e_i] = -Optim.minimum(res_nd_patient)
                else
                    @inbounds @views V_nd[a_i,:,e_i] .= utility_function(0.0,σ)
                end
            end

            # compute value
            Threads.@threads for a_i in 1:a_size
                @inbounds CoH = w*exp(e) + a_grid[a_i]
                if CoH >= 0.0
                    @inbounds @views V[a_i,:,e_i] = V_nd[a_i,:,e_i]
                else
                    @inbounds @views V[a_i,:,e_i] = V_d[:,e_i]
                end
            end
        end

        # check convergence
        crit = norm(V.-V_p, Inf)

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update iteration number
        iter += 1
    end

    return p, q, V, V_d, V_nd
end

function variables_function(
    parameters::NamedTuple
    )
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack a_size, a_size_μ, e_size, ν_size = parameters

    # initialization
    p, q, V, V_d, V_nd = solve_ED_function(parameters; tol = 1E-8, iter_max = 1000)
    policy_a = zeros(a_size, ν_size, e_size)
    policy_d = zeros(a_size, ν_size, e_size)

    # define the type distribution and its transition matrix
    μ_size = a_size_μ*e_size*ν_size
    μ = ones(a_size_μ, e_size, ν_size)./μ_size

    # define aggregate variables
    K = 0.0
    L = 0.0
    N = 0.0
    aggregate_var = MutableAggregateVariables(K, L, N)

    # return outputs
    variables = MutableVariables(V, V_d, V_nd, policy_a, policy_d, p, q, μ, aggregate_var)
    return variables
end

function value_function!(
    V_p::Array{Float64,3},
    q_p::Array{Float64,2},
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )
    """
    update value and policy functions
    """

    # unpack parameters
    @unpack a_grid, a_size, e_grid, e_size, e_Γ, β_grid, β_size, β_Γ = parameters
    @unpack action_grid, action_ind, a_ind_zero, action_size, s_grid, s_size, κ_grid, ρ, γ, α = parameters

    # feasible set and conditional value function
    Threads.@threads for e_i in 1:e_size

        @inbounds e = e_grid[e_i]
        @inbounds ν = ν_grid[β_i]

        # compute the next-period discounted expected value funtions and interpolated functions
        @inbounds @views V_expt_p = (ν_Γ[1]*V_p[:,:,1] + ν_Γ[2]*V_p[:,:,2])*e_Γ[e_i,:]
        @inbounds @views V_hat_impatient = ν_grid[1]*β*V_expt_p
        @inbounds @views V_hat_patient = ν_grid[2]*β*V_expt_p
        V_hat_impatient_itp = Akima(a_grid, V_hat_impatient)
        V_hat_patient_itp = Akima(a_grid, V_hat_patient)

        # compute defaulting value
        variables.V_d[e_i,1] = utility_function((1-η)*w*exp(e),σ) + V_hat_impatient[a_ind_zero]
        variables.V_d[e_i,2] = utility_function((1-η)*w*exp(e),σ) + V_hat_patient[a_ind_zero]

        q = q_p[:,e_i]
        qa = [q.*a_grid_neg; a_grid_pos[2:end]]
        qa_itp = Akima(a_grid, qa)



        if c <= 0.0 || (action_i == 1 && a_i >= a_ind_zero)
            @inbounds variables.v[action_i,β_i,e_i,a_i,s_i] = -Inf
        else
            W_expect = 0.0
            for e_p_i in 1:e_size, β_p_i in 1:β_size
                if action_i == 1
                    @inbounds W_expect += β_Γ[β_i,β_p_i]*e_Γ[e_i,e_p_i]*W_p[β_p_i,e_p_i,a_p_i,1]
                else
                    for s_p_i in 1:s_size
                        @inbounds W_expect += β_Γ[β_i,β_p_i]*e_Γ[e_i,e_p_i]*variables.Q_s[s_p_i,action_i,e_i,a_i,s_i]*W_p[β_p_i,e_p_i,a_p_i,s_p_i]
                    end
                end
            end
            @inbounds variables.v[action_i,β_i,e_i,a_i,s_i] = utility_function(c,γ) + β*ρ*W_expect
        end
    end

    # unconditional value function and choice probability
    Threads.@threads for s_i in 1:s_size
        for a_i in 1:a_size, e_i in 1:e_size, β_i in 1:β_size
            @inbounds @views V = sum(exp.(variables.v[:,β_i,e_i,a_i,s_i]./α))
            @inbounds variables.W[β_i,e_i,a_i,s_i] = α*log(V)
            @inbounds @views variables.σ[:,β_i,e_i,a_i,s_i] = exp.(variables.v[:,β_i,e_i,a_i,s_i]./α)./V
            @inbounds @views variables.F[:,β_i,e_i,a_i,s_i] .= 1.0
            @inbounds @views variables.F[variables.σ[:,β_i,e_i,a_i,s_i].≈0.0,β_i,e_i,a_i,s_i] .= 0.0

            @inbounds @views V_ND = sum(exp.(variables.v[2:end,β_i,e_i,a_i,s_i]./α))
            if V_ND ≈ 0.0
                @inbounds @views variables.σ_ND[:,β_i,e_i,a_i,s_i] .= 0.0
            else
                @inbounds @views variables.σ_ND[:,β_i,e_i,a_i,s_i] = exp.(variables.v[2:end,β_i,e_i,a_i,s_i]./α)./V_ND
            end
            @inbounds @views variables.F_ND[:,β_i,e_i,a_i,s_i] .= 1.0
            @inbounds @views variables.F_ND[variables.σ_ND[:,β_i,e_i,a_i,s_i].≈0.0,β_i,e_i,a_i,s_i] .= 0.0
        end
    end

    if slow_updating != 1.0
        variables.W = slow_updating*variables.W + (1.0-slow_updating)*W_p
    end
end

#=================#
# Solve the model #
#=================#
parameters = parameters_function()
variables = variables_function(parameters)

e_label = round.(exp.(parameters.e_grid),digits=2)'
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg,:],legend=:bottomright,label=e_label)
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg,:].*parameters.a_grid_neg,legend=:bottomright,label=e_label)
