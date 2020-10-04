function para_func(;
    ψ_H::Real = 0.10,           # history rased probability
    ψ_B::Real = 0.80,           # bank's survival rate
    β::Real = 0.96,             # discount factor
    ξ::Real = 0.30,             # garnishment rate
    σ::Real = 2,                # CRRA coefficient
    z::Real = 1,                # aggregate uncertainty
    LR::Real = 5,               # targeted leverage ratio
    r_ld::Real = 0.01,          # targeted excess return
    r_d::Real = 0.03,           # deposit rate
    e_ρ::Real = 0.95,           # AR(1) of earnings shock
    e_σ::Real = 0.20,           # s.d. of earnings shock
    e_size::Integer = 5,        # no. of earnings shock
    ν_size::Integer = 2,        # no. of preference shock
    ν_s::Real = 0.70,           # scale of patience
    ν_p::Real = 0.05,           # probability of patience
    a_min::Real = -0.1,         # min of asset holding
    a_max::Real = 50,           # max of asset holding
    a_size::Integer = 80,       # number of the grid asset holding for VFI
    a_size_μ::Integer = 2000    # number of the grid asset holding for μ
    )
    #-----------------------------------------------------#
    # contruct an immutable object containg all paramters #
    #-----------------------------------------------------#

    # persistent shock
    e_M = rouwenhorst(e_size, e_ρ, e_σ)
    e_Γ = e_M.p
    e_grid = exp.(collect(e_M.state_values))

    # preference schock
    ν_grid = [ν_s, 1]
    ν_Γ = repeat([ν_p 1-ν_p], ν_size, 1)

    # idiosyncratic transition matrix
    x_Γ = kron(ν_Γ, e_Γ)
    x_grid = gridmake(e_grid, ν_grid)
    x_ind = gridmake(1:e_size, 1:ν_size)
    x_size = e_size*ν_size

    # asset holding grid for VFI
    a_size_neg = convert(Int, a_size/2)
    a_grid_neg = collect(range(a_min, 0, length = a_size_neg))
    a_size_pos = a_size - a_size_neg + 1
    a_grid_pos = collect(range(0, a_max, length = a_size_pos))
    a_grid = cat(a_grid_neg, a_grid_pos[2:end], dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(a_grid .== 0)[1]

    # asset holding grid for μ
    a_size_neg_μ = convert(Int, a_size_μ/2)
    a_grid_neg_μ = collect(range(a_min, 0, length = a_size_neg_μ))
    a_size_pos_μ = a_size_μ - a_size_neg_μ + 1
    a_grid_pos_μ = collect(range(0, a_max, length = a_size_pos_μ))
    a_grid_μ = cat(a_grid_neg_μ, a_grid_pos_μ[2:end], dims = 1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(a_grid_μ .== 0)[1]

    # solve the steady state of ω and θ to match targeted parameters
    θ = (β*(1-ψ_B)) / ((1-β*ψ_B)*LR)
    Λ = β*(1 - ψ_B + ψ_B*θ*LR)
    λ = (Λ*r_ld) / θ
    # ω = (1 - ψ_B*(r_ld*LR+(1+r_d))) / ((1-ψ_B)*(1+r_d+r_ld)*LR)

    # return values
    return (ψ_H = ψ_H, ψ_B = ψ_B, β = β, ξ = ξ, σ = σ, z = z,
            LR = LR, r_ld = r_ld, r_d = r_d, θ = θ, #ω = ω,
            a_grid = a_grid, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos,
            a_size = a_size, a_size_neg = a_size_neg, a_size_pos = a_size_pos,
            a_ind_zero = a_ind_zero,
            a_grid_μ = a_grid_μ, a_grid_neg_μ = a_grid_neg_μ, a_grid_pos_μ = a_grid_pos_μ,
            a_size_μ = a_size_μ, a_size_neg_μ = a_size_neg_μ, a_size_pos_μ = a_size_pos_μ,
            a_ind_zero_μ = a_ind_zero_μ,
            e_Γ = e_Γ, e_grid = e_grid, e_size = e_size,
            ν_Γ = ν_Γ, ν_grid = ν_grid, ν_size = ν_size,
            x_Γ = x_Γ, x_grid = x_grid, x_size = x_size, x_ind = x_ind)
end

mutable struct mut_var
    q::Array{Float64,2}
    V_good::Array{Float64,2}
    V_good_repay::Array{Float64,2}
    V_good_default::Array{Float64,2}
    V_bad::Array{Float64,2}
    policy_a_good::Array{Float64,2}
    policy_a_bad::Array{Float64,2}
    μ_good::Array{Float64,2}
    μ_bad::Array{Float64,2}
    aggregate_var::Array{Float64,1}
end

function var_func(parameters::NamedTuple)
    #------------------------------------------------------------#
    # construct a mutable object containing endogenous variables #
    #------------------------------------------------------------#

    # unpack parameters
    @unpack a_grid, a_size, a_size_pos = parameters
    @unpack a_size_μ, a_size_pos_μ = parameters
    @unpack x_size = parameters
    @unpack r_d, r_ld = parameters

    # define pricing related variables
    q = ones(a_size, x_size)
    q[findall(a_grid .< 0),:] .= 1 / (1 + r_d + r_ld)

    # define value functions
    V_good = zeros(a_size, x_size)
    V_good_repay = zeros(a_size, x_size)
    V_good_default = zeros(a_size, x_size)
    V_bad = zeros(a_size_pos, x_size)

    # define policy functions
    policy_a_good = zeros(a_size, x_size)
    policy_a_bad = zeros(a_size_pos, x_size)

    # define the type distribution and its transition matrix
    μ_size = x_size*(a_size_μ + a_size_pos_μ)
    μ_good = ones(a_size_μ,x_size) ./ μ_size
    μ_bad = ones(a_size_pos_μ,x_size) ./ μ_size

    # define aggregate objects
    aggregate_var = zeros(4)

    # return outputs
    variables = mut_var(q, V_good, V_good_repay, V_good_default, V_bad, policy_a_good, policy_a_bad, μ_good, μ_bad, aggregate_var)
    return variables
end

function u_func(c::Real, σ::Real)
    #--------------------------------------------------------------#
    # compute utility of CRRA utility function with coefficient σ. #
    #--------------------------------------------------------------#
    if c > 0
        return σ == 1 ? log(c) : 1 / ((1-σ)*c^(σ-1))
    else
        return -Inf
    end
end

function mu_inv_func(mu::Real, σ::Real)
    #-----------------------------------------------------------------------------#
    # compute the inverse of utility of CRRA utility function with coefficient σ. #
    #-----------------------------------------------------------------------------#
    if mu > 0
        return 1/(mu^(1/σ))
    else
        return 10^18
    end
end

function qa_itp_piecewise(
    a_p::Real,
    a_grid::Array{Float64,1},
    q::Array{Float64,1},
    r_d::Real,
    r_ld::Real
    )

    a_ind_norisk = findall(q .>= 1/(1+r_d+r_ld))
    a_ind_risk = findall(q .< 1/(1+r_d+r_ld))

    a_grid_norisk = a_grid[a_ind_norisk]
    a_grid_risk = cat(a_grid[a_ind_risk], a_grid_norisk[1], dims = 1)

    q_norisk = q[a_ind_norisk]
    q_risk = cat(q[a_ind_risk], 1/(1+r_d+r_ld), dims = 1)

    qa_norisk = q_norisk .* a_grid_norisk
    qa_risk = q_risk .* a_grid_risk

    if a_p < a_grid_norisk[1]
        qa_risk_itp = Spline1D(a_grid_risk, qa_risk, k = 1)
        qa = qa_risk_itp(a_p)
    else
        qa_norisk_itp = Spline1D(a_grid_norisk, qa_norisk, k = 1)
        qa = qa_norisk_itp(a_p)
    end

    return qa
end

function value_func!(
    V_good_p::Array{Float64,2},
    V_bad_p::Array{Float64,2},
    q_p::Array{Float64,2},
    variables::mut_var,
    parameters::NamedTuple
    )

    @unpack a_grid, a_grid_pos, a_ind_zero = parameters
    @unpack a_size, a_size_pos = parameters
    @unpack x_grid, x_size, x_ind, x_Γ = parameters
    @unpack e_grid, e_Γ = parameters
    @unpack β, σ, ξ, ψ_H, r_d, r_ld = parameters

    for x_i in 1:x_size

        e, ν = x_grid[x_i,:]
        q = q_p[:,x_i]
        # object_qa(a_p) = qa_itp_piecewise(a_p, a_grid, q, r_d, r_ld)
        qa_itp = LinearInterpolation(a_grid, a_grid .* q, extrapolation_bc = Line())

        V_hat_bad = (ν*β)*(ψ_H*V_good_p[a_ind_zero:end,:].+(1-ψ_H)*V_bad_p)*x_Γ[x_i,:]
        V_hat_good = (ν*β)*V_good_p*x_Γ[x_i,:]

        V_hat_bad_itp = LinearInterpolation(a_grid_pos, V_hat_bad, extrapolation_bc = Line())
        V_hat_good_itp = LinearInterpolation(a_grid, V_hat_good, extrapolation_bc = Line())

        # bad credit history
        for a_i in 1:a_size_pos
            a = a_grid_pos[a_i]
            object_bad(a_p) = -(u_func((1-ξ)*e+(1+r_d)*a-a_p, σ) + V_hat_bad_itp(a_p))
            res_bad = optimize(object_bad, 0.0, (1-ξ)*e+(1+r_d)*a)
            variables.V_bad[a_i,x_i] = -Optim.minimum(res_bad)
            variables.policy_a_bad[a_i,x_i] = Optim.minimizer(res_bad)
        end

        # good credit history and default
        variables.V_good_default[:,x_i] .= u_func(e, σ) + V_hat_bad[1]

        # good credit history and repay
        for a_i in 1:a_size
            a = a_grid[a_i]
            # res_rbl = optimize(object_qa, a_grid[1], 0.0)
            # object_good_repay(a_p) = -(u_func(e+(1+r_d*(a>0.0))*a-object_qa(a_p), σ) + V_hat_good_itp(a_p))
            # res_good_repay = optimize(object_good_repay, Optim.minimizer(res_rbl), a_grid[end])
            object_good_repay(a_p) = -(u_func(e+(1+r_d*(a>0.0))*a-qa_itp(a_p), σ) + V_hat_good_itp(a_p))
            res_good_repay = optimize(object_good_repay, a_grid[1], a_grid[end])
            variables.V_good_repay[a_i,x_i] = -Optim.minimum(res_good_repay)

            if variables.V_good_default[a_i,x_i] > variables.V_good_repay[a_i,x_i]
                variables.V_good[a_i,x_i] = variables.V_good_default[a_i,x_i]
                variables.policy_a_good[a_i,x_i] = 0.0
            else
                variables.V_good[a_i,x_i] = variables.V_good_repay[a_i,x_i]
                variables.policy_a_good[a_i,x_i] = Optim.minimizer(res_good_repay)
            end
        end
    end
end

function price_func!(
    q_p::Array{Float64,2},
    variables::mut_var,
    parameters::NamedTuple
    )
    #-------------------------------------------------------#
    # update the price schedule and associated derivatives. #
    #-------------------------------------------------------#
    @unpack r_d, r_ld = parameters
    @unpack a_size, a_grid_neg, a_size_neg = parameters
    @unpack x_size, x_ind, x_Γ = parameters
    α = 0.7    # parameter controling update speed
    q_update = ones(a_size, x_size)

    for x_i in 1:x_size
        for a_p_i in 1:a_size_neg
            revenue = 0
            if a_p_i < a_size_neg
                for x_p_i in 1:x_size
                    if variables.V_good_default[a_p_i,x_p_i] < variables.V_good_repay[a_p_i,x_p_i]
                        revenue += x_Γ[x_i,x_p_i]
                    end
                end
                q_update[a_p_i,x_i] = revenue/(1+r_d+r_ld)
            end
        end
    end

    variables.q = α*q_update + (1-α)*q_p
end

function solve_func!(
    variables::mut_var,
    parameters::NamedTuple;
    tol = 1E-6,
    iter_max = 10000
    )

    # unpack parameters
    # @unpack a_grid, a_size, a_grid_neg, a_grid_pos, ind_a_zero, x_grid, x_size, β, Px, λ_H, σ, ξ, r_f, z = parameters

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving household's maximization: ")

    # initialize the next-period value functions
    V_good_p = similar(variables.V_good)
    V_bad_p = similar(variables.V_bad)
    q_p = similar(variables.q)

    while crit > tol && iter < iter_max

        # println("iter = $iter")
        # copy the current value functions to the pre-specified containers
        copyto!(V_good_p, variables.V_good)
        copyto!(V_bad_p, variables.V_bad)
        copyto!(q_p, variables.q)

        # update value function
        value_func!(V_good_p, V_bad_p, q_p, variables, parameters)

        # update price, its derivative, and size of bond
        price_func!(q_p, variables, parameters)

        # check convergence
        crit = max(norm(variables.V_good[:,:,1]-V_good_p, Inf), norm(variables.V_bad-V_bad_p, Inf), norm(variables.q-q_p, Inf))

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end

    # update the cross-sectional distribution
    # LoM_func!(variables, parameters)

    # compute aggregate variables
    # aggregate_func!(variables, parameters)
    # println("The lower state of preference shock is $(parameters.ν_grid[1])")
    # println("The excess return is $(parameters.r_bf)")
    # println("The risk-free rate is $(parameters.r_f)")
    # println("Targeted leverage ratio is $(parameters.L) and the implied leverage ratio is $(variables.A[4])")
    # ED = variables.A[1] - (parameters.L/(parameters.L-1))*variables.A[2]
    # println("Excess demand is $ED")
    # return ED
end

using LinearAlgebra
using ProgressMeter
using Parameters
using QuantEcon: rouwenhorst, gridmake
using Plots
using Optim
# using Dierckx
using Interpolations

parameters = para_func()
variables = var_func(parameters)

solve_func!(variables, parameters)
