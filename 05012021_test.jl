using Parameters
using LinearAlgebra
using ProgressMeter
using Plots
using Colors
using LaTeXStrings

function parameters_FI_function(;
    β_L::Real = 0.886,                          # low type discount factor
    β_H::Real = 0.915,                          # high type discount factor
    Prob_β_L_to_H::Real = 0.013,                # transition from low to high β
    Prob_β_H_to_L::Real = 0.011,                # transition from high to low β
    κ::Real = 0.02,                             # lump-sum filing cost
    γ::Real = 3.00,                             # risk aversion
    r::Real = 0.01,                             # risk-free rate
    ρ::Real = 0.975,                            # survival probability
    α::Real = 3.387*0.001,                      # EV scale parameter
    λ::Real = 0.991,                            # EV correlation parameters
    a_size_neg::Integer = 51,                   # number of negative assets
    a_size_pos::Integer = 101,                  # number of positive assets
    a_min::Real = -0.25,                        # minimum of assets
    a_max::Real = 15.00,                        # maximum of assets
    )
    """
    contruct an immutable object containg all paramters
    """

    # discount factor
    β_grid = [β_L, β_H]
    β_size = length(β_grid)
    β_Γ = [1-Prob_β_L_to_H   Prob_β_L_to_H;
             Prob_β_H_to_L 1-Prob_β_H_to_L]

    # persistent earnings
    e_grid = [0.57, 1.00, 1.74]
    e_size = length(e_grid)
    e_Γ = [0.818  0.178  0.004;
           0.178  0.644  0.178;
           0.004  0.178  0.818]

    # transitory earnings
    z_grid = [-0.18, 0.00, 0.18]
    z_size = length(z_grid)
    z_Γ = [1/3, 1/3, 1/3]

    # asset holdings
    a_grid_neg = collect(range(a_min, 0.0, length = a_size_neg))
    a_grid_pos = collect(range(0.0, a_max, length = a_size_pos))
    a_grid = cat(a_grid_neg, a_grid_pos[2:end], dims = 1)
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

    # searching set of binary monotonicity
    BM_indices = BM_function(a_size)

    # criterion
    # L_λα = (λ*α)*log(eps(Float64))
    L_λα = (λ*α)*log(0.0)

    # return the outcome
    return (κ = κ, γ = γ, r = r, ρ = ρ, α = α, λ = λ,
            β_grid = β_grid, β_size = β_size, β_Γ = β_Γ,
            e_grid = e_grid, e_size = e_size, e_Γ = e_Γ,
            z_grid = z_grid, z_size = z_size, z_Γ = z_Γ,
            a_grid_neg = a_grid_neg, a_size_neg = a_size_neg,
            a_grid_pos = a_grid_pos, a_size_pos = a_size_pos,
            a_grid = a_grid, a_size = a_size, a_ind_zero = a_ind_zero,
            action_grid = action_grid, action_ind = action_ind,
            action_size = action_size,
            BM_indices = BM_indices, L_λα = L_λα)
end

mutable struct MutableVariables_FI
    """
    construct a type for mutable variables
    """
    v::Array{Float64,5}
    W::Array{Float64,4}
    F::Array{Float64,5}
    σ::Array{Float64,5}
    P::Array{Float64,3}
    q::Array{Float64,3}
    μ::Array{Float64,4}
end

function variables_FI_function(
    parameters_FI::NamedTuple
    )
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack action_size, β_size, z_size, e_size, a_size, r, ρ = parameters_FI

    # (un)conditional value functions
    v = zeros(action_size, β_size, z_size, e_size, a_size)
    W = zeros(β_size, z_size, e_size, a_size)

    # feasible set
    F = zeros(action_size, β_size, z_size, e_size, a_size)

    # choice probability
    σ = ones(action_size, β_size, z_size, e_size, a_size) ./ action_size

    # repayment probability and loan pricing
    P = ones(a_size, β_size, e_size)
    q = ones(a_size, β_size, e_size) .* (ρ/(1.0 + r))

    # cross-sectional distribution
    μ = ones(β_size, z_size, e_size, a_size) ./ (β_size*z_size*e_size*a_size)

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
    W_p::Array{Float64,4},
    variables_FI::MutableVariables_FI,
    parameters_FI::NamedTuple;
    GQ_algorithm::Bool = false
    )
    """
    compute feasible set and (un)conditional value functions
    """

    @unpack a_grid, a_size, a_ind_zero, e_grid, e_size, e_Γ, z_grid, z_size, z_Γ = parameters_FI
    @unpack β_grid, β_size, β_Γ = parameters_FI
    @unpack action_grid, action_ind, action_size, κ, ρ, γ, λ, α = parameters_FI
    @unpack BM_indices, L_λα = parameters_FI

    if GQ_algorithm == false

        # feasible set and conditional value function
        for a_i in 1:a_size, e_i in 1:e_size, z_i in 1:z_size, β_i in 1:β_size, action_i in 1:action_size

            # retrieve the associated states
            a = a_grid[a_i]
            e = e_grid[e_i]
            z = z_grid[z_i]
            d, a_p = action_grid[action_i,:]
            d_i, a_p_i = action_ind[action_i,:]

            # compute consumption IN Equation (2)
            if d == 0.0
                c = e + z + a - variables_FI.q[a_p_i,β_i,e_i]*a_p
            else
                c = e + z - κ
            end

            # check feasibility
            if c > 0.0
                variables_FI.F[action_i,β_i,z_i,e_i,a_i] = 1.0
            else
                variables_FI.F[action_i,β_i,z_i,e_i,a_i] = 0.0
            end

            # update conditional value function in Equation (4)
            variables_FI.v[action_i,β_i,z_i,e_i,a_i] = Π_function(c, a_p_i, β_i, e_i, W_p, parameters_FI)
        end

        # unconditional value function
        for a_i in 1:a_size, e_i in 1:e_size, z_i in 1:z_size, β_i in 1:β_size

            # retrieve asset holding
            a = a_grid[a_i]

            # compute summation over feasible set in Equation (9)
            sum_Eq9 = sum(variables_FI.F[2:end,β_i,z_i,e_i,a_i] .* exp.(variables_FI.v[2:end,β_i,z_i,e_i,a_i]./(λ*α)))

            # compute expected value of not defaulting in Equation (9)
            W_ND = α*log(sum_Eq9)

            # compute first term above in Equation (10)
            first_Eq10 = exp(variables_FI.v[1,β_i,z_i,e_i,a_i]/α)

            # compute expected value function in Equation (10)
            if a < 0.0
                variables_FI.W[β_i,z_i,e_i,a_i] = α*log(first_Eq10 + exp(λ*W_ND/α))
            else
                variables_FI.W[β_i,z_i,e_i,a_i] = α*log(exp(λ*W_ND/α))
            end
        end

    else

        # initialize conditional value fuinction
        variables_FI.v .= -Inf

        # feasible set and conditional value function
        for e_i in 1:e_size, z_i in 1:z_size, β_i in 1:β_size
            # println("e_i = $e_i and z_i = $z_i and β_i = $β_i")
            # retrieve the associated states
            e = e_grid[e_i]
            z = z_grid[z_i]

            # defaulting value
            c = e + z - κ
            variables_FI.v[1,β_i,z_i,e_i,:] .= Π_function(c, a_ind_zero, β_i, e_i, W_p, parameters_FI)

            # non-defaulting value
            c_ND(a_i, a_p_i) = e + z + a_grid[a_i] - variables_FI.q[a_p_i,β_i,e_i]*a_grid[a_p_i]
            Π_ND(a_i, a_p_i) = Π_function(c_ND(a_i,a_p_i), a_p_i, β_i, e_i, W_p, parameters_FI)

            BM_bounds = zeros(Int,a_size,3)
            BM_bounds[:,1] = 1:a_size
            BM_bounds[1,2] = 1
            BM_bounds[1,3] = a_size
            BM_bounds[end,3] = a_size

            for BM_i in 1:a_size

                # retrieve the associated states
                a_i, lb_i, ub_i = BM_indices[BM_i,:]
                lb = BM_bounds[lb_i,2]
                ub = BM_bounds[ub_i,3]
                Π(a_p_i) = Π_ND(a_i, a_p_i)

                # compute optimal choice and associated value
                i_p_star, U_star = HM_algorithm(lb, ub, Π)

                # if all choices are infeasible
                if U_star == -Inf
                    # update new bounds
                    BM_bounds[a_i,2] = 1
                    BM_bounds[a_i,3] = 1

                # if aby choice is feasible
                else
                    # assign the optimal value
                    variables_FI.v[1+i_p_star,β_i,z_i,e_i,a_i] = U_star

                    # check new lower bound
                    lb_star = i_p_star - 1
                    flag_lb = 0
                    while (lb <= lb_star) && (flag_lb == 0)
                        variables_FI.v[1+lb_star,β_i,z_i,e_i,a_i] = Π(lb_star)
                        if variables_FI.v[1+lb_star,β_i,z_i,e_i,a_i] - U_star >= L_λα
                            lb_star -= 1
                        else
                            flag_lb = 1
                        end
                    end

                    # check new upper bound
                    ub_star = i_p_star + 1
                    flag_ub = 0
                    while (ub >= ub_star) && (flag_ub == 0)
                        variables_FI.v[1+ub_star,β_i,z_i,e_i,a_i] = Π(ub_star)
                        if variables_FI.v[1+ub_star,β_i,z_i,e_i,a_i] - U_star >= L_λα
                            ub_star += 1
                        else
                            flag_ub = 1
                        end
                    end

                    # update new bounds
                    BM_bounds[a_i,2] = lb_star + 1
                    BM_bounds[a_i,3] = ub_star - 1
                end
            end
        end

        # unconditional value function
        for a_i in 1:a_size, e_i in 1:e_size, z_i in 1:z_size, β_i in 1:β_size

            # retrieve asset holding
            a = a_grid[a_i]

            # compute summation over feasible set in Equation (9)
            sum_Eq9 = sum(variables_FI.F[2:end,β_i,z_i,e_i,a_i] .* exp.(variables_FI.v[2:end,β_i,z_i,e_i,a_i]./(λ*α)))

            # compute expected value of not defaulting in Equation (9)
            W_ND = α*log(sum_Eq9)

            # compute first term above in Equation (10)
            first_Eq10 = exp(variables_FI.v[1,β_i,z_i,e_i,a_i]/α)

            # compute expected value function in Equation (10)
            if a < 0.0
                variables_FI.W[β_i,z_i,e_i,a_i] = α*log(first_Eq10 + exp(λ*W_ND/α))
            else
                variables_FI.W[β_i,z_i,e_i,a_i] = α*log(exp(λ*W_ND/α))
            end
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

    @unpack a_grid, a_size, e_grid, e_size, e_Γ, z_grid, z_size, z_Γ, β_grid, β_size, β_Γ = parameters_FI
    @unpack action_grid, action_ind, action_size, λ, α = parameters_FI

    for a_i in 1:a_size, e_i in 1:e_size, z_i in 1:z_size, β_i in 1:β_size

        # retrieve asset holding
        a = a_grid[a_i]

        # compute summation over feasible set in the denominator in Equation (6)
        # den_sum_Eq6 = sum(variables_FI.F[2:end,β_i,z_i,e_i,a_i] .* exp.(variables_FI.v[2:end,β_i,z_i,e_i,a_i]./(λ*α)))
        den_sum_Eq6 = sum(exp.(variables_FI.v[2:end,β_i,z_i,e_i,a_i]./(λ*α)))

        # compute numerator in Equation (6)
        num_Eq6 = exp(variables_FI.v[1,β_i,z_i,e_i,a_i]/α)

        # compute probability of default in Equation (6)
        if a < 0.0
            variables_FI.σ[1,β_i,z_i,e_i,a_i] = num_Eq6 / (num_Eq6 + den_sum_Eq6^λ)
        else
            variables_FI.σ[1,β_i,z_i,e_i,a_i] = 0.0
        end

        # compute choice probability
        for action_i in 2:action_size

            if variables_FI.v[action_i,β_i,z_i,e_i,a_i] != -Inf
                # compute the numerator in Equation (7)
                num_Eq7 = exp(variables_FI.v[action_i,β_i,z_i,e_i,a_i]/(λ*α))

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
            variables_FI.σ[action_i,β_i,z_i,e_i,a_i] = σ_tilde*(1-variables_FI.σ[1,β_i,z_i,e_i,a_i])
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

    @unpack a_grid, a_size, e_grid, e_size, e_Γ, z_grid, z_size, z_Γ, β_grid, β_size, β_Γ = parameters_FI
    @unpack action_grid, action_ind, action_size, r, ρ = parameters_FI

    # store previous repaying probability
    P_p = similar(variables_FI.P)
    copyto!(P_p, variables_FI.P)

    for e_i in 1:e_size, β_i in 1:β_size, a_p_i in 1:a_size

        # retrieve asset holding in next period
        a_p = a_grid[a_p_i]

        # compute probability of repayment with full information in Equation (31)
        P_expect = 0.0
        for e_p_i in 1:e_size, z_p_i in 1:z_size, β_p_i in 1:β_size
            P_expect += (1.0-variables_FI.σ[1,β_p_i,z_p_i,e_p_i,a_p_i])*β_Γ[β_i,β_p_i]*e_Γ[e_i,e_p_i]*z_Γ[z_p_i]
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
    one_loop::Bool = true,
    GQ_algorithm::Bool = false
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
            value_function!(W_p, variables_FI, parameters_FI; GQ_algorithm = GQ_algorithm)

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
                value_function!(W_p, variables_FI, parameters_FI; GQ_algorithm = GQ_algorithm)

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

function Π_function(
    c::Real,
    a_p_i::Integer,
    β_i::Integer,
    e_i::Integer,
    W_p::Array{Float64,4},
    parameters_FI::NamedTuple;
    u::Function = utility_function
    )
    """
    compute conditional value function in Equation (4)
    """

    @unpack e_grid, e_size, e_Γ, z_grid, z_size, z_Γ, β_grid, β_size, β_Γ = parameters_FI
    @unpack γ, ρ = parameters_FI

    # retrieve deiscount factor
    β = β_grid[β_i]

    # compute expected unconditional value
    W_expect = 0.0
    for e_p_i in 1:e_size, z_p_i in 1:z_size, β_p_i in 1:β_size
        W_expect += β_Γ[β_i,β_p_i]*e_Γ[e_i,e_p_i]*z_Γ[z_p_i]*W_p[β_p_i,z_p_i,e_p_i,a_p_i]
    end

    # return result
    return (1.0-β*ρ)*u(c, γ) + β*ρ*W_expect
end

function BM_function(
    action_size::Integer
    )
    """
    compute the sorted indices of binary search algorithm

    BM_index[:,1] saves the middle point
    BM_index[:,2] saves lower bound
    BM_index[:,3] saves upper bound
    """

    # initialize auxiliary matrix
    auxiliary_matrix = []
    push!(auxiliary_matrix, [1, action_size])

    # initialize the matrix storing binary search indices
    BM_index = zeros(Int, 2, 3)
    BM_index[1,:] = [1 1 1]
    BM_index[2,:] = [action_size 1 action_size]

    # set up criterion and iteration number
    k = Inf
    iter = 1

    while k > 1

        # initializa the number of rows, i.e., k
        if iter == 1
            k = 1
        end

        # step 2 on page 35 in Gordon and Qiu (2017, WP)
        while (auxiliary_matrix[end][1]+1) < auxiliary_matrix[end][2]
            m = convert(Int, floor((auxiliary_matrix[end][1]+auxiliary_matrix[end][2])/2))

            if findall(BM_index[:,1] .== m) == []
                BM_index = cat(BM_index, [m auxiliary_matrix[end][1] auxiliary_matrix[end][2]]; dims = 1)
            end

            push!(auxiliary_matrix, [auxiliary_matrix[end][1], m])
            k += 1
        end

        # step 3 on page 35 in Gordon and Qiu (2017, WP)
        if k == 1
            break
        end
        while auxiliary_matrix[end][2] == auxiliary_matrix[end-1][2]
            pop!(auxiliary_matrix)
            k -= 1
            if k == 1
                break
            end
        end
        if k == 1
            break
        end

        # step 4 on page 35 in Gordon and Qiu (2017, WP)
        auxiliary_matrix[end][1] = auxiliary_matrix[end][2]
        auxiliary_matrix[end][2] = auxiliary_matrix[end-1][2]

        # update iteration number
        iter += 1
    end

    # return results
    return BM_index
end

function HM_algorithm(
    lb::Integer,            # lower bound of choices
    ub::Integer,            # upper bound of choices
    Π::Function             # return function
    )
    """
    implement Heer and Maussner's (2005) algorithm of binary concavity
    """

    while true
        # points of considered
        n = ub - lb + 1

        # step 1 on page 536 in Gordon and Qiu (2018, QE)
        if n == 1
            return lb, Π(lb)
        else
            flag_lb = 0
            flag_ub = 0
        end

        # step 2 on page 536 in Gordon and Qiu (2018, QE)
        if n == 2
            if flag_lb == 0
                Π_lb = Π(lb)
            end
            if flag_ub == 0
                Π_ub = Π(ub)
            end
            if Π_lb > Π_ub
                return lb, Π_lb
            else
                return ub, Π_ub
            end
        end

        # step 3 on page 536 in Gordon and Qiu (2018, QE)
        if n == 3
            if max(flag_lb, flag_ub) == 0
                Π_lb = Π(lb)
                flag_lb = 1
            end
            m = convert(Int, (lb+ub)/2)
            Π_m = Π(m)
            if flag_lb == 1
                if Π_lb > Π_m
                    return  lb, Π_lb
                end
                lb, Π_lb, flag_lb = m, Π_m, 1
            else # flag_ub == 1
                if Π_ub > Π_m
                    return  ub, Π_ub
                end
                ub, Π_ub, flag_ub = m, Π_m, 1
            end
        end

        # step 4 on page 536 in Gordon and Qiu (2018, QE)
        m = convert(Int, floor((lb+ub)/2))
        Π_m = Π(m)
        Π_m1 = Π(m+1)
        if Π_m < Π_m1
            lb, Π_lb, flag_lb = (m+1), Π_m1, 1
        else
            ub, Π_ub, flag_ub = m, Π_m, 1
        end
    end
end

parameters_FI = parameters_FI_function()
variables_FI = variables_FI_function(parameters_FI)
solve_function!(variables_FI, parameters_FI; tol = 1E-8, iter_max = 1000, one_loop = true, GQ_algorithm = true)

# check whether the sum of choice probability, given any individual state, equals one
#all(sum(variables_FI.σ, dims=1) .≈ 1.0)

# plot the loan pricing function of low β
label_latex = reshape(latexstring.("\$",["e = $(parameters_FI.e_grid[i])" for i in 1:parameters_FI.e_size],"\$"),1,:)
plot(parameters_FI.a_grid_neg, variables_FI.q[1:parameters_FI.a_ind_zero,1,:],
     label = label_latex, legend = :topleft, legendfont = font(10))


#=
# heatmap for checking monotonicity
heatmap(parameters_FI.a_grid, parameters_FI.a_grid, variables_FI.v[2:end,1,2,2,:]')

mcols = [:cornsilk, :antiquewhite, :navajowhite, :navajowhite3, :navajowhite4]
iter = 1
PS = []
for a_i_x in 1:parameters_FI.a_size
    x = parameters_FI.a_grid[a_i_x]
    if maximum(variables_FI.v[2:end,1,2,2,a_i_x]) > -Inf
        a_i_y_optimal = argmax(variables_FI.v[2:end,1,2,2,a_i_x])
    else
        y = parameters_FI.a_grid[1]
        if iter == 1
            PS = scatter([x], [y], markercolor = :red, markerstrokecolor = :auto, legend = :none)
        else
            scatter!([x], [y], markercolor = :red, markerstrokecolor = :auto, legend = :none)
        end
        iter += 1
    end
    for a_i_y in 1:parameters_FI.a_size
        println("x_i = $a_i_x and y_i = $a_i_y")
        if variables_FI.v[1+a_i_y,1,2,2,a_i_x] != -Inf
            y = parameters_FI.a_grid[a_i_y]
            if -Inf < variables_FI.v[1+a_i_y,1,2,2,a_i_x] <= -0.70
                k = mcols[1]
            elseif -0.70 < variables_FI.v[1+a_i_y,1,2,2,a_i_x] <= -0.50
                k = mcols[2]
            elseif -0.50 < variables_FI.v[1+a_i_y,1,2,2,a_i_x] <= -0.30
                k = mcols[3]
            elseif -0.30 < variables_FI.v[1+a_i_y,1,2,2,a_i_x] <= -0.10
                k = mcols[4]
            else
                k = mcols[5]
            end
            if iter == 1
                PS = scatter([x], [y], markercolor = k, markerstrokecolor = :auto, legend = :none)
            else
                scatter!([x], [y], markercolor = k, markerstrokecolor = :auto, legend = :none)
            end
            iter += 1
            if a_i_y == a_i_y_optimal
                scatter!([x], [y], markercolor = :red, markerstrokecolor = :auto, legend = :none)
            end
        end
    end
end
PS
=#
