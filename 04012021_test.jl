using Parameters
using LinearAlgebra
using ProgressMeter
using Plots
using LaTeXStrings
using QuantEcon: rouwenhorst, tauchen, gridmake, MarkovChain, stationary_distributions

function parameters_FI_function(;
    β_H::Real = 0.96,                           # discount factor (households)
    β_s::Real = 0.95,                           # scale of impatience
    β_p::Real = 0.10,                           # probability of impatience
    β_B::Real = 0.96,                           # discount factor (banks)
    κ::Real = 0.40,                             # proportional filing cost
    γ::Real = 2.00,                             # risk aversion
    ρ::Real = 1.00,                             # survival probability
    σ::Real = 0.005,                            # EV scale parameter
    δ::Real = 0.08,                             # depreciation rate
    α::Real = 0.33,                             # capital share
    K2Y::Real = 3.0,                            # capital to output ratio
    ψ::Real = 0.95,                             # bank's survival rate
    λ::Real = 0.00,                             # multiplier of incentive constraint
    θ::Real = 0.40,                             # diverting fraction
    e_ρ::Real = 0.95,                           # AR(1) of persistent earnings
    e_σ::Real = 0.10,                           # s.d. of persistent earnings
    e_size::Integer = 3,                        # number of persistent earnings
    a_size_neg::Integer = 101,                  # number of negative assets
    a_size_pos::Integer = 51,                   # number of positive assets
    a_min::Real = -5.00,                        # minimum of assets
    a_max::Real = 50.00,                        # maximum of assets
    a_degree::Integer = 1,                      # curvature of negative asset gridpoints
    μ_scale::Integer = 7                        # scale governing the number of grids in computing density
    )
    """
    contruct an immutable object containg all paramters
    """

    # discount factor
    β_grid = [β_H*β_s, β_H]
    β_size = length(β_grid)
    β_Γ = [β_p, 1.0-β_p]

    # persistent earnings
    e_MC = tauchen(e_size, e_ρ, e_σ, 0.0, 3)
    e_Γ = e_MC.p
    e_grid = exp.(collect(e_MC.state_values))
    e_SD = stationary_distributions(e_MC)[]
    e_SS = sum(e_SD .* e_grid)

    # asset holdings
    a_grid_neg = reverse(((range(0.0, stop = a_size_neg-1.0, length = a_size_neg)/(a_size_neg-1)).^a_degree)*a_min)
    a_grid_pos = collect(range(0.0, a_max, length = a_size_pos))
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero, a_grid)[]

    # asset holdings for μ
    a_size_neg_μ = convert(Int, (a_size_neg-1)*μ_scale+1)
    a_grid_neg_μ = collect(range(a_min, 0.0, length = a_size_neg_μ))
    a_size_pos_μ = convert(Int, (a_size_pos-1)*μ_scale+1)
    a_grid_pos_μ = collect(range(0.0, a_max, length = a_size_pos_μ))
    a_grid_μ = cat(a_grid_neg_μ[1:(end-1)], a_grid_pos_μ, dims = 1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(iszero, a_grid_μ)[]

    # action
    action_grid = zeros(a_size+1, 2)
    action_grid[1,1] = 1.0
    action_grid[2:end,2] = a_grid
    action_ind = zeros(Int64, a_size+1, 2)
    action_ind[1,1] = 1
    action_ind[1,2] = a_ind_zero
    action_ind[2:end,1] .= 2
    action_ind[2:end,2] = collect(1:a_size)
    action_size = size(action_grid)[1]

    # compute equilibrium prices and quantities
    i = (α/K2Y) - δ
    ξ = (β_B*(1.0-ψ)*(1.0+i)) / ((1.0-λ)-β_B*ψ*(1.0+i))
    Λ = β_B*(1.0-ψ+ψ*ξ)
    LR = ξ/θ
    AD = LR/(LR-1.0)
    r_lp = λ*θ/Λ
    r_k = i + r_lp
    K = exp(e_SS)*(α/(r_k+δ))^(1.0/(1.0-α))
    w = (1.0-α)*(K^α)*(exp(e_SS)^(-α))

    # return the outcome
    return (β_B, κ = κ, γ = γ, ρ = ρ, σ = σ, δ = δ, α = α, K2Y = K2Y,
            ψ = ψ, λ = λ, θ = θ, i = i, ξ = ξ, Λ = Λ, LR = LR, AD = AD,
            r_lp = r_lp, r_k = r_k, K = K, w = w,
            β_grid = β_grid, β_size = β_size, β_Γ = β_Γ,
            e_grid = e_grid, e_size = e_size, e_Γ = e_Γ,
            a_grid = a_grid, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos,
            a_size = a_size, a_size_neg = a_size_neg, a_size_pos = a_size_pos,
            a_ind_zero = a_ind_zero,
            a_grid_μ = a_grid_μ, a_grid_neg_μ = a_grid_neg_μ, a_grid_pos_μ = a_grid_pos_μ,
            a_size_μ = a_size_μ, a_size_neg_μ = a_size_neg_μ, a_size_pos_μ = a_size_pos_μ,
            a_ind_zero_μ = a_ind_zero_μ,
            action_grid = action_grid, action_ind = action_ind,
            action_size = action_size)
end

mutable struct MutableVariables_FI
    """
    construct a type for mutable variables
    """
    v::Array{Float64,4}
    W::Array{Float64,3}
    F::Array{Float64,4}
    σ::Array{Float64,4}
    P::Array{Float64,2}
    q::Array{Float64,2}
    μ::Array{Float64,3}
end

function variables_FI_function(
    parameters_FI::NamedTuple
    )
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack action_size, β_size, e_size, a_size, a_size_neg, ρ, i, r_lp = parameters_FI

    # (un)conditional value functions
    v = zeros(action_size, β_size, e_size, a_size)
    W = zeros(β_size, e_size, a_size)

    # feasible set
    F = zeros(action_size, β_size, e_size, a_size)

    # choice probability
    σ = ones(action_size, β_size, e_size, a_size) ./ action_size

    # repayment probability and loan pricing
    P = ones(a_size_neg, e_size)
    q = ones(a_size_neg, e_size) .* (ρ/(1.0 + i + r_lp))

    # cross-sectional distribution
    μ = ones(β_size, e_size, a_size) ./ (β_size*e_size*a_size)

    # return the outcome
    variables_FI = MutableVariables_FI(v, W, F, σ, P, q, μ)
    return variables_FI
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

function value_function!(
    W_p::Array{Float64,3},
    q_p::Array{Float64,2},
    variables_FI::MutableVariables_FI,
    parameters_FI::NamedTuple
    )
    """
    compute feasible set and (un)conditional value functions
    """

    @unpack a_grid, a_grid_neg, a_grid_pos, a_size, e_grid, e_size, e_Γ, β_grid, β_size, β_Γ = parameters_FI
    @unpack action_grid, action_ind, action_size, κ, ρ, γ, σ, i, w = parameters_FI

    # feasible set and conditional value function
    for a_i in 1:a_size, e_i in 1:e_size, β_i in 1:β_size, action_i in 1:action_size
        # retrieve the associated states
        a = a_grid[a_i]
        e = e_grid[e_i]
        β = β_grid[β_i]
        d, a_p = action_grid[action_i,:]
        d_i, a_p_i = action_ind[action_i,:]

        # construct the size of asset holdings
        q = q_p[:,e_i]
        qa = [q.*a_grid_neg; a_grid_pos[2:end]]

        # compute consumption
        if d == 0.0
            c = w*e + a*(1.0+i*(a>0.0)) - qa[a_p_i]
        else
            c = w*e*(1.0 - κ)
        end

        # check feasibility
        if c > 0.0
            variables_FI.F[action_i,β_i,e_i,a_i] = 1.0
        else
            variables_FI.F[action_i,β_i,e_i,a_i] = 0.0
        end

        # update conditional value function
        W_expect = 0.0
        for e_p_i in 1:e_size, β_p_i in 1:β_size
            W_expect += β_Γ[β_p_i]*e_Γ[e_i,e_p_i]*W_p[β_p_i,e_p_i,a_p_i]
        end
        variables_FI.v[action_i,β_i,e_i,a_i] = (1.0-β*ρ)*utility_function(c, γ) + β*ρ*W_expect
    end

    # unconditional value function
    for a_i in 1:a_size, e_i in 1:e_size, β_i in 1:β_size
        # compute summation over feasible set
        sum_all = sum(variables_FI.F[:,β_i,e_i,a_i] .* exp.(variables_FI.v[:,β_i,e_i,a_i]./σ))
        variables_FI.W[β_i,e_i,a_i] = σ*log(sum_all)
    end
end

function sigma_function!(
    variables_FI::MutableVariables_FI,
    parameters_FI::NamedTuple
    )
    """
    compute choice probability
    """

    @unpack a_grid, a_size, e_grid, e_size, e_Γ, β_grid, β_size, β_Γ = parameters_FI
    @unpack action_grid, action_ind, action_size, σ = parameters_FI

    for a_i in 1:a_size, e_i in 1:e_size, β_i in 1:β_size, action_i in 1:action_size

        # retrieve asset holding
        a = a_grid[a_i]

        # compute summation over feasible set in the denominator
        sum_all = sum(variables_FI.F[:,β_i,e_i,a_i] .* exp.(variables_FI.v[:,β_i,e_i,a_i]./σ))

        # compute numerator
        num = exp(variables_FI.v[action_i,β_i,e_i,a_i]/σ)

        # compute choice probability
        if variables_FI.F[action_i,β_i,e_i,a_i] != 0.0
            if num ≈ 0.0
                variables_FI.σ[action_i,β_i,e_i,a_i] = 0.0
            else
                variables_FI.σ[action_i,β_i,e_i,a_i] = num / sum_all
            end
        else
            variables_FI.σ[action_i,β_i,e_i,a_i] = 0.0
        end
    end
end

function loan_price_function!(
    q_p::Array{Float64,2},
    variables_FI::MutableVariables_FI,
    parameters_FI::NamedTuple;
    slow_updating::Real = 1.0
    )
    """
    compute repaying probability and loan price
    """

    @unpack a_grid, a_size_neg, e_grid, e_size, e_Γ, β_grid, β_size, β_Γ = parameters_FI
    @unpack action_grid, action_ind, action_size, i, r_lp, ρ = parameters_FI

    # store previous repaying probability
    q_temp = similar(variables_FI.q)

    # compute repayment probability
    for e_i in 1:e_size, a_p_i in 1:a_size_neg
        P_expect = 0.0
        for e_p_i in 1:e_size, β_p_i in 1:β_size
            P_expect += (1.0-variables_FI.σ[1,β_p_i,e_p_i,a_p_i])*β_Γ[β_p_i]*e_Γ[e_i,e_p_i]
        end
        variables_FI.P[a_p_i,e_i] = P_expect
    end

    # update loan pricing function
    q_temp = ρ*variables_FI.P./(1.0+i+r_lp)
    variables_FI.q = slow_updating*q_temp + (1.0-slow_updating)*q_p
end

function solve_function!(
    variables_FI::MutableVariables_FI,
    parameters_FI::NamedTuple;
    tol::Real = tol_h,
    iter_max::Integer = iter_max,
    one_loop::Bool = true
    )
    """
    solve model
    """

    if one_loop == true

        # initialize the iteration number and criterion
        iter = 0
        crit = Inf
        prog = ProgressThresh(tol, "Solving model (one-loop): ")

        # initialize the next-period functions
        W_p = similar(variables_FI.W)
        q_p = similar(variables_FI.q)

        while crit > tol && iter < iter_max

            # copy previous unconditional value and loan pricing functions
            copyto!(W_p, variables_FI.W)
            copyto!(q_p, variables_FI.q)

            # update value functions
            value_function!(W_p, q_p, variables_FI, parameters_FI)

            # compute choice probability
            sigma_function!(variables_FI, parameters_FI)

            # compute loan price
            loan_price_function!(q_p, variables_FI, parameters_FI; slow_updating = 1.0)

            # check convergence
            crit = max(norm(variables_FI.W .- W_p, Inf), norm(variables_FI.q .- q_p, Inf))

            # report preogress
            ProgressMeter.update!(prog, crit)

            # update the iteration number
            iter += 1
        end

    else

        # initialize the iteration number and criterion
        iter_q = 0
        crit_q = Inf
        prog_q = ProgressThresh(tol, "Solving loan pricing function: ")

        # initialize the next-period function
        q_p = similar(variables_FI.q)

        while crit_q > tol && iter_q < iter_max

            # copy previous loan pricing function
            copyto!(q_p, variables_FI.q)

            # initialize the iteration number and criterion
            iter_W = 0
            crit_W = Inf
            prog_W = ProgressThresh(tol, "Solving unconditional value function: ")

            # initialize the next-period function
            W_p = similar(variables_FI.W)

            while crit_W > tol && iter_W < iter_max

                # copy previous loan pricing function
                copyto!(W_p, variables_FI.W)

                # update value functions
                value_function!(W_p, q_p, variables_FI, parameters_FI)

                # check convergence
                crit_W = norm(variables_FI.W .- W_p, Inf)

                # report preogress
                ProgressMeter.update!(prog_W, crit_W)

                # update the iteration number
                iter_W += 1

            end

            # compute choice probability
            sigma_function!(variables_FI, parameters_FI)

            # compute loan price
            loan_price_function!(variables_FI, parameters_FI; slow_updating = 1.0)

            # check convergence
            crit_q = norm(variables_FI.q .- q_p, Inf)

            # report preogress
            ProgressMeter.update!(prog_q, crit_q)

            # update the iteration number
            iter_q += 1
            println("")
        end
    end
end

# solve the model
parameters_FI = parameters_FI_function()
variables_FI = variables_FI_function(parameters_FI)
solve_function!(variables_FI, parameters_FI; tol = 1E-6, iter_max = 1000, one_loop = true)

# check whether the sum of choice probability, given any individual state, equals one
all(sum(variables_FI.σ, dims=1) .≈ 1.0)

# plot the loan pricing function of low β
label_latex = reshape(latexstring.("\$",["e = $(round(parameters_FI.e_grid[i],digits=2))" for i in 1:parameters_FI.e_size],"\$"),1,:)
title_latex = latexstring("\$","\\kappa = $(parameters_FI.κ)","\$")
plot(parameters_FI.a_grid_neg, variables_FI.q,
     title = title_latex, label = label_latex, legend = :bottomright, legendfont = font(6))

#=
plot(parameters_FI.a_grid_neg[end-50:end], variables_FI.q[parameters_FI.a_ind_zero-50:parameters_FI.a_ind_zero,1,:],
     label = label_latex, legend = :none, legendfont = font(10), seriestype=:scatter)

plot(parameters_FI.a_grid_neg, -parameters_FI.a_grid_neg.*variables_FI.q[1:parameters_FI.a_ind_zero,1,:],
     label = label_latex, legend = :none, legendfont = font(10), seriestype=:scatter)

plot(parameters_FI.a_grid, variables_FI.σ[2:end,1,:,180], legend=:none)
=#

#=
P_test = similar(variables_FI.P)

@unpack a_grid, a_size, e_grid, e_size, e_Γ, β_grid, β_size, β_Γ = parameters_FI
@unpack action_grid, action_ind, action_size, r, ρ = parameters_FI

for e_i in 1:e_size, β_i in 1:β_size, a_p_i in 1:a_size

    P_expect = 0.0

    for e_p_i in 1:e_size, β_p_i in 1:β_size
        P_expect += (1.0-variables_FI.σ[1,β_p_i,e_p_i,a_p_i])*β_Γ[β_i,β_p_i]*e_Γ[e_i,e_p_i]
    end

    P_test[a_p_i,β_i,e_i] = P_expect

end

a_degree = 2.0
a_min = -5.0

a_size_neg_1 = 81
a_size_neg_1_adj = a_size_neg_1 - 1
a_grid_neg_1 = reverse(((range(0.0, stop = a_size_neg_1_adj-1.0, length = a_size_neg_1_adj)/(a_size_neg_1_adj-1)).^a_degree)*a_min)

a_thres_2 = -1.0
a_saved_2 = 10
a_size_neg_2 = a_size_neg_1 - a_saved_2
a_min_2 = a_min - a_thres_2
a_grid_neg_2 = reverse(((range(0.0, stop = a_size_neg_2-1.0, length = a_size_neg_2)/(a_size_neg_2-1)).^a_degree)*a_min_2.+a_thres_2)
a_grid_neg_2 = cat(a_grid_neg_2[1:(end-1)], collect(range(a_thres_2, stop = 0.0, length = a_saved_2)); dims =1)

a_thres_3 = -2.0
a_saved_3 = 10
a_size_neg_3 = a_size_neg_1 - a_saved_3
a_min_3 = a_min - a_thres_3
a_grid_neg_3 = reverse(((range(0.0, stop = a_size_neg_3-1.0, length = a_size_neg_3)/(a_size_neg_3-1)).^a_degree)*a_min_3.+a_thres_3)
a_grid_neg_3 = cat(a_grid_neg_3[1:(end-1)], collect(range(a_thres_3, stop = 0.0, length = a_saved_3)); dims =1)

plot(a_grid_neg_1, repeat([3], outer=a_size_neg_1_adj), seriestype = :scatter)
plot!(a_grid_neg_2, repeat([2], outer=a_size_neg_1_adj), seriestype = :scatter)
plot!(a_grid_neg_3, repeat([1], outer=a_size_neg_1_adj), seriestype = :scatter, legend = :none)
=#

# a_grid_neg = cat(a_grid_neg_1, a_grid_neg_2, a_grid_neg_3; dims = 2)
# legend_text = ["save 0 points" "save 10 points" "save 20 points"]
# plot(a_grid_neg, label = legend_text, legend = :bottomright, seriestype = :scatter)
