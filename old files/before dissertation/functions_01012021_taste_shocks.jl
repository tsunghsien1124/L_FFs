using Parameters
using LinearAlgebra
using ProgressMeter
using Plots
using LaTeXStrings
using QuantEcon: rouwenhorst, tauchen, gridmake, MarkovChain, stationary_distributions

function parameters_FI_function(;
    β_L::Real = 0.886,                          # low type discount factor
    β_H::Real = 0.915,                          # high type discount factor
    Prob_β_L_to_H::Real = 0.013,                # transition from low to high β
    Prob_β_H_to_L::Real = 0.011,                # transition from high to low β
    κ::Real = 0.60,                             # proportional filing cost
    γ::Real = 2.00,                             # risk aversion
    r::Real = 0.01,                             # risk-free rate
    ρ::Real = 0.975,                            # survival probability
    α::Real = 0.005,                            # EV scale parameter
    λ::Real = 0.991,                            # EV correlation parameters
    δ::Real = 0.08,                             # depreciation rate
    # α::Real = 0.33,                             # capital share
    K2Y::Real = 3.0,                            # capital to output ratio
    ψ::Real = 0.95,                             # bank's survival rate
    # λ::Real = 0.00,                             # multiplier of incentive constraint
    θ::Real = 0.40,                             # diverting fraction
    e_ρ::Real = 0.95,                           # AR(1) of persistent earnings
    e_σ::Real = 0.15,                           # s.d. of persistent earnings
    e_size::Integer = 11,                       # number of persistent earnings
    a_size_neg::Integer = 151,                  # number of negative assets
    a_size_pos::Integer = 51,                   # number of positive assets
    a_min::Real = -5.00,                        # minimum of assets
    a_max::Real = 500.00,                       # maximum of assets
    a_degree::Integer = 2,                      # curvature of negative asset gridpoints
    )
    """
    contruct an immutable object containg all paramters
    """

    # discount factor
    β_grid = [β_L, β_H]
    β_size = length(β_grid)
    β_Γ = [1.0-Prob_β_L_to_H     Prob_β_L_to_H;
               Prob_β_H_to_L 1.0-Prob_β_H_to_L]

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

    # return the outcome
    return (κ = κ, γ = γ, r = r, ρ = ρ, α = α, λ = λ,
            β_grid = β_grid, β_size = β_size, β_Γ = β_Γ,
            e_grid = e_grid, e_size = e_size, e_Γ = e_Γ,
            a_grid_neg = a_grid_neg, a_size_neg = a_size_neg,
            a_grid_pos = a_grid_pos, a_size_pos = a_size_pos,
            a_grid = a_grid, a_size = a_size, a_ind_zero = a_ind_zero,
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
    P::Array{Float64,3}
    q::Array{Float64,3}
    μ::Array{Float64,3}
end

function variables_FI_function(
    parameters_FI::NamedTuple
    )
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack action_size, β_size, e_size, a_size, r, ρ = parameters_FI

    # (un)conditional value functions
    v = zeros(action_size, β_size, e_size, a_size)
    W = zeros(β_size, e_size, a_size)

    # feasible set
    F = zeros(action_size, β_size, e_size, a_size)

    # choice probability
    σ = ones(action_size, β_size, e_size, a_size) ./ action_size

    # repayment probability and loan pricing
    P = ones(a_size, β_size, e_size)
    q = ones(a_size, β_size, e_size) .* (ρ/(1.0 + r))

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
    variables_FI::MutableVariables_FI,
    parameters_FI::NamedTuple
    )
    """
    compute feasible set and (un)conditional value functions
    """

    @unpack a_grid, a_size, e_grid, e_size, e_Γ, β_grid, β_size, β_Γ = parameters_FI
    @unpack action_grid, action_ind, action_size, κ, ρ, γ, λ, α = parameters_FI

    # feasible set and conditional value function
    for a_i in 1:a_size, e_i in 1:e_size, β_i in 1:β_size, action_i in 1:action_size

        # retrieve the associated states
        a = a_grid[a_i]
        e = e_grid[e_i]
        β = β_grid[β_i]
        d, a_p = action_grid[action_i,:]
        d_i, a_p_i = action_ind[action_i,:]

        # compute consumption IN Equation (2)
        if d == 0.0
            c = e  + a - variables_FI.q[a_p_i,β_i,e_i]*a_p
        else
            # c = e + z - κ
            c = e*(1.0 - κ)
        end

        # check feasibility
        if c > 0.0
            variables_FI.F[action_i,β_i,e_i,a_i] = 1.0
        else
            variables_FI.F[action_i,β_i,e_i,a_i] = 0.0
        end

        # update conditional value function in Equation (4)
        W_expect = 0.0
        for e_p_i in 1:e_size, β_p_i in 1:β_size
            W_expect += β_Γ[β_i,β_p_i]*e_Γ[e_i,e_p_i]*W_p[β_p_i,e_p_i,a_p_i]
        end
        variables_FI.v[action_i,β_i,e_i,a_i] = (1.0-β*ρ)*utility_function(c, γ) + β*ρ*W_expect
    end

    # unconditional value function
    for a_i in 1:a_size, e_i in 1:e_size, β_i in 1:β_size

        # retrieve asset holding
        a = a_grid[a_i]

        # compute summation over feasible set in Equation (9)
        sum_Eq9 = sum(variables_FI.F[2:end,β_i,e_i,a_i] .* exp.(variables_FI.v[2:end,β_i,e_i,a_i]./(λ*α)))

        # compute expected value of not defaulting in Equation (9)
        W_ND = α*log(sum_Eq9)

        # compute first term above in Equation (10)
        first_Eq10 = exp(variables_FI.v[1,β_i,e_i,a_i]/α)

        # compute expected value function in Equation (10)
        if a < 0.0
            variables_FI.W[β_i,e_i,a_i] = α*log(first_Eq10 + exp(λ*W_ND/α))
        else
            variables_FI.W[β_i,e_i,a_i] = α*log(exp(λ*W_ND/α))
        end
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
    @unpack action_grid, action_ind, action_size, λ, α = parameters_FI

    for a_i in 1:a_size, e_i in 1:e_size, β_i in 1:β_size

        # retrieve asset holding
        a = a_grid[a_i]

        # compute summation over feasible set in the denominator in Equation (6)
        den_sum_Eq6 = sum(variables_FI.F[2:end,β_i,e_i,a_i] .* exp.(variables_FI.v[2:end,β_i,e_i,a_i]./(λ*α)))

        # compute numerator in Equation (6)
        num_Eq6 = exp(variables_FI.v[1,β_i,e_i,a_i]/α)

        # compute probability of default in Equation (6)
        if a < 0.0
            variables_FI.σ[1,β_i,e_i,a_i] = num_Eq6 / (num_Eq6 + den_sum_Eq6^λ)
        else
            variables_FI.σ[1,β_i,e_i,a_i] = 0.0
        end

        # compute choice probability
        for action_i in 2:action_size

            if variables_FI.F[action_i,β_i,e_i,a_i] != 0.0
                # compute the numerator in Equation (7)
                num_Eq7 = exp(variables_FI.v[action_i,β_i,e_i,a_i]/(λ*α))

                # compute choice probability conditional on not defaulting in Equation (7)
                if num_Eq7 ≈ 0.0
                    σ_tilde = 0.0
                else
                    σ_tilde = num_Eq7 / den_sum_Eq6
                end
            else
                σ_tilde = 0.0
            end

            # compute unconditional probability in Equation (8)
            variables_FI.σ[action_i,β_i,e_i,a_i] = σ_tilde*(1-variables_FI.σ[1,β_i,e_i,a_i])
        end
    end
end

function loan_price_function!(
    variables_FI::MutableVariables_FI,
    parameters_FI::NamedTuple;
    slow_updating::Real = 1.0
    )
    """
    compute repaying probability and loan price
    """

    @unpack a_grid, a_size, e_grid, e_size, e_Γ, β_grid, β_size, β_Γ = parameters_FI
    @unpack action_grid, action_ind, action_size, r, ρ = parameters_FI

    # store previous repaying probability
    P_p = similar(variables_FI.P)
    copyto!(P_p, variables_FI.P)

    for e_i in 1:e_size, β_i in 1:β_size, a_p_i in 1:a_size

        # retrieve asset holding in next period
        a_p = a_grid[a_p_i]

        # compute probability of repayment with full information in Equation (31)
        P_expect = 0.0
        for e_p_i in 1:e_size, β_p_i in 1:β_size
            P_expect += (1.0-variables_FI.σ[1,β_p_i,e_p_i,a_p_i])*β_Γ[β_i,β_p_i]*e_Γ[e_i,e_p_i]
        end
        variables_FI.P[a_p_i,β_i,e_i] = slow_updating*P_expect + (1.0-slow_updating)*P_p[a_p_i,β_i,e_i]

        # compute loan price in Equation (12)
        if a_p < 0.0
            variables_FI.q[a_p_i,β_i,e_i] = ρ*variables_FI.P[a_p_i,β_i,e_i]/(1+r)
        else
            variables_FI.q[a_p_i,β_i,e_i] = ρ/(1.0+r)
        end
    end
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
            value_function!(W_p, variables_FI, parameters_FI)

            # compute choice probability
            sigma_function!(variables_FI, parameters_FI)

            # compute loan price
            loan_price_function!(variables_FI, parameters_FI; slow_updating = 1.0)

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
                value_function!(W_p, variables_FI, parameters_FI)

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

parameters_FI = parameters_FI_function()
variables_FI = variables_FI_function(parameters_FI)
solve_function!(variables_FI, parameters_FI; tol = 1E-8, iter_max = 1000, one_loop = true)

# check whether the sum of choice probability, given any individual state, equals one
all(sum(variables_FI.σ, dims=1) .≈ 1.0)

# plot the loan pricing function of low β
label_latex = reshape(latexstring.("\$",["e = $(parameters_FI.e_grid[i])" for i in 1:parameters_FI.e_size],"\$"),1,:)

plot(parameters_FI.a_grid_neg, variables_FI.q[1:parameters_FI.a_ind_zero,1,:],
     label = label_latex, legend = :none, legendfont = font(10))

#=
plot(parameters_FI.a_grid_neg[end-50:end], variables_FI.q[parameters_FI.a_ind_zero-50:parameters_FI.a_ind_zero,1,:],
     label = label_latex, legend = :none, legendfont = font(10), seriestype=:scatter)

plot(parameters_FI.a_grid_neg, -parameters_FI.a_grid_neg.*variables_FI.q[1:parameters_FI.a_ind_zero,1,:],
     label = label_latex, legend = :none, legendfont = font(10), seriestype=:scatter)

plot(parameters_FI.a_grid, variables_FI.σ[2:end,1,:,180], legend=:none)
=#
