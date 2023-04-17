# include("SimplePCHIP.jl")
# include("FLOWMath.jl")

# using Main.FLOWMath: interp2d, akima
# using Main.SimplePCHIP
using LinearAlgebra
using ProgressMeter
using Parameters
using QuantEcon: rouwenhorst, gridmake
using Plots
using Optim
using PrettyTables
using Distributions
using Roots
using Dierckx
# using Interpolations


function para_func(;
    ψ_H::Real = 0.10,           # history rased probability
    ψ_B::Real = 0.80,           # bank's survival rate
    β::Real = 0.96,             # discount factor (households)
    β_B::Real = 0.90,           # discount factor (banks)
    ξ::Real = 0.35,             # garnishment rate
    σ::Real = 2,                # CRRA coefficient
    z::Real = 1,                # aggregate uncertainty
    r_d::Real = 0.02,           # deposit rate
    λ::Real = 0.00,             # the multiplier of incentive constraint
    θ::Real = 0.20,             # the diverted fraction
    e_ρ::Real = 0.95,           # AR(1) of earnings shock
    e_σ::Real = 0.10,           # s.d. of earni ngs shock
    e_size::Integer = 21,       # no. of earnings shock
    ν_size::Integer = 2,        # no. of preference shock
    ν_s::Real = 0.80,           # scale of patience
    ν_p::Real = 0.10,           # probability of patience
    a_min::Real = -3,           # min of asset holding
    a_max::Real = 20,           # max of asset holding
    a_size_neg::Integer = 41,   # number of the grid of negative asset holding for VFI
    a_size_pos::Integer = 41,   # number of the grid of positive asset holding for VFI
    a_degree::Integer = 2,      # curvature of the positive asset gridpoints
    μ_scale::Integer = 5        # scale governing the number of grids in computing density
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
    a_grid_neg = collect(range(a_min, 0, length = a_size_neg))
    a_grid_pos = ((range(0, stop = a_size_pos-1, length = a_size_pos)/(a_size_pos-1)).^a_degree)*a_max
    a_grid = cat(a_grid_neg, a_grid_pos[2:end], dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(a_grid .== 0)[1]

    # asset holding grid for μ
    a_size_neg_μ = convert(Int, a_size_neg*μ_scale)
    a_grid_neg_μ = collect(range(a_min, 0, length = a_size_neg_μ))
    a_size_pos_μ = convert(Int, a_size_pos*μ_scale)
    a_grid_pos_μ = collect(range(0, a_max, length = a_size_pos_μ))
    a_grid_μ = cat(a_grid_neg_μ, a_grid_pos_μ[2:end], dims = 1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(a_grid_μ .== 0)[1]

    # solve the steady state of ω and θ to match targeted parameters
    # λ = 1 - β*ψ_B*(1+r_d) - 10^(-4)
    # λ = 1 - (β*ψ_B*(1+r_d))^(1/2)
    α = (β_B*(1-ψ_B)*(1+r_d)) / ((1-λ)-β_B*ψ_B*(1+r_d))
    Λ = β_B*(1-ψ_B+ψ_B*α)
    r_ld = λ*θ/Λ
    LR = α/θ
    ω = ((1-ψ_B)^(-1)) * (((1+r_d+r_ld)*LR-(1+r_d))^(-1) - ψ_B)

    # return values
    return (ψ_H = ψ_H, ψ_B = ψ_B, β = β, β_B = β_B, ξ = ξ, σ = σ, z = z,
            LR = LR, r_ld = r_ld, r_d = r_d, θ = θ, λ = λ, α = α, Λ = Λ, ω = ω,
            a_grid = a_grid, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos,
            a_size = a_size, a_size_neg = a_size_neg, a_size_pos = a_size_pos,
            a_ind_zero = a_ind_zero,
            a_grid_μ = a_grid_μ, a_grid_neg_μ = a_grid_neg_μ, a_grid_pos_μ = a_grid_pos_μ,
            a_size_μ = a_size_μ, a_size_neg_μ = a_size_neg_μ, a_size_pos_μ = a_size_pos_μ,
            a_ind_zero_μ = a_ind_zero_μ, μ_scale = μ_scale,
            e_ρ = e_ρ, e_σ = e_σ, e_Γ = e_Γ, e_grid = e_grid, e_size = e_size,
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
    policy_d_good::Array{Float64,2}
    μ_good::Array{Float64,2}
    μ_bad::Array{Float64,2}
    aggregate_var::Array{Float64,1}
end

function var_func(parameters::NamedTuple)
    #------------------------------------------------------------#
    # construct a mutable object containing endogenous variables #
    #------------------------------------------------------------#

    # unpack parameters
    @unpack a_grid, a_size, a_size_pos, a_ind_zero = parameters
    @unpack a_size_μ, a_size_pos_μ = parameters
    @unpack x_size, ν_size, e_grid = parameters
    @unpack σ, r_d, r_ld = parameters

    # define pricing related variables
    q = ones(a_size, x_size)
    q[findall(a_grid .< 0),:] .= 1 / (1 + r_d + r_ld)

    # define value functions
    V_good = zeros(a_size, x_size)
    V_good_repay = zeros(a_size, x_size)
    V_good_default = zeros(a_size, x_size)
    V_bad = zeros(a_size_pos, x_size)

    # V_good = u_func.(repeat(e_grid',a_size,ν_size) .+ repeat((1 .+ r_d*(a_grid.>0)).*a_grid,1,x_size), σ)
    # copyto!(V_bad, V_good[a_ind_zero:end,:])

    # define policy functions
    policy_a_good = zeros(a_size, x_size)
    policy_a_bad = zeros(a_size_pos, x_size)
    policy_d_good = zeros(a_size, x_size)

    # define the type distribution and its transition matrix
    μ_size = x_size*(a_size_μ + a_size_pos_μ)
    μ_good = ones(a_size_μ,x_size) ./ μ_size
    μ_bad = ones(a_size_pos_μ,x_size) ./ μ_size

    # define aggregate objects
    aggregate_var = zeros(4)

    # return outputs
    variables = mut_var(q, V_good, V_good_repay, V_good_default, V_bad, policy_a_good, policy_a_bad, policy_d_good, μ_good, μ_bad, aggregate_var)
    return variables
end

function u_func(c::Real, σ::Real)
    #--------------------------------------------------------------#
    # compute utility of CRRA utility function with coefficient σ. #
    #--------------------------------------------------------------#
    if c > 0
        return σ == 1 ? log(c) : 1 / ((1-σ)*c^(σ-1))
    else
        return -10^8
    end
end

function mu_inverse_func(mu::Real, σ::Real)
    #--------------------------------------------------------------#
    # compute utility of CRRA utility function with coefficient σ. #
    #--------------------------------------------------------------#
    if mu > 0
        return 1/(mu^(1/σ))
    else
        return Inf
    end
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

        # println("x_i = $x_i")
        e, ν = x_grid[x_i,:]
        q = q_p[:,x_i]
        qa_itp = Spline1D(a_grid, q .* a_grid, k = 1, bc = "extrapolate")
        object_rbl(a_p) = qa_itp(a_p)
        res_rbl = optimize(object_rbl, a_grid[1], 0)
        rbl = Optim.minimizer(res_rbl)

        V_hat_bad = (ν*β)*(ψ_H*V_good_p[a_ind_zero:end,:].+(1-ψ_H)*V_bad_p)*x_Γ[x_i,:]
        V_hat_good = (ν*β)*V_good_p*x_Γ[x_i,:]
        V_hat_bad_itp = Spline1D(a_grid_pos, V_hat_bad, k = 1, bc = "extrapolate")
        V_hat_good_itp = Spline1D(a_grid, V_hat_good, k = 1, bc = "extrapolate")

        # bad credit history
        for a_i in 1:a_size_pos
            a = a_grid_pos[a_i]
            CoH = e + (1+r_d)*a
            object_bad(a_p) = -(u_func(CoH - a_p, σ) + V_hat_bad_itp(a_p))
            res_bad = optimize(object_bad, 0, min(a_grid[end],CoH))
            variables.V_bad[a_i,x_i] = -Optim.minimum(res_bad)
            variables.policy_a_bad[a_i,x_i] = Optim.minimizer(res_bad)
        end

        # good credit history and default
        variables.V_good_default[:,x_i] .= u_func((1-ξ)*e, σ) + V_hat_bad[1]

        # good credit history and repay
        for a_i in 1:a_size
            # println("a_i = $a_i")

            a = a_grid[a_i]
            CoH = e + (1+r_d*(a>0))*a
            object_good(a_p) = -(u_func(CoH - qa_itp(a_p), σ) + V_hat_good_itp(a_p))
            res_good = optimize(object_good, a_grid[1], min(a_grid[end],CoH))
            variables.V_good_repay[a_i,x_i] = -Optim.minimum(res_good)
            if variables.V_good_default[a_i,x_i] > variables.V_good_repay[a_i,x_i]
                variables.V_good[a_i,x_i] = variables.V_good_default[a_i,x_i]
                variables.policy_a_good[a_i,x_i] = 0.0
                variables.policy_d_good[a_i,x_i] = 1.0
            else
                variables.V_good[a_i,x_i] = variables.V_good_repay[a_i,x_i]
                variables.policy_a_good[a_i,x_i] = Optim.minimizer(res_good)
                variables.policy_d_good[a_i,x_i] = 0.0
            end
        end
    end
end

function price_func!(
    q_p::Array{Float64,2},
    variables::mut_var,
    parameters::NamedTuple
    )
    #----------------------------#
    # update the price schedule. #
    #----------------------------#
    @unpack ξ, r_d, r_ld = parameters
    @unpack a_size, a_grid, a_grid_neg, a_size_neg = parameters
    @unpack x_ind, x_size, x_grid, x_Γ = parameters
    @unpack e_size, e_grid, e_ρ, e_σ = parameters

    Δ = 0.5    # parameter controling update speed

    a_grid_neg_nozero = a_grid_neg[1:(end-1)]
    a_size_neg_nozero = length(a_grid_neg_nozero)
    q_update = ones(a_size_neg_nozero, x_size)

    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        ν_lower = (ν_i-1)*e_size + 1
        ν_upper = ν_i*e_size
        V_good_diff = variables.V_good_repay[:,ν_lower:ν_upper] .- variables.V_good_default[:,ν_lower:ν_upper]
        dist = LogNormal(e_ρ*e_grid[e_i], e_σ)
        for a_p_i in 1:a_size_neg_nozero
            println("$x_i, $a_p_i")
            V_diff_itp = Spline1D(e_grid, V_good_diff[a_p_i,:], k = 1, bc = "extrapolate")
            object_V_diff(x_p) = V_diff_itp(x_p)
            x_threshold = find_zero(object_V_diff, e_grid[end])
            default_prob = cdf(dist, x_threshold)
            q_update[a_p_i,x_i] = (1-default_prob) / (1+r_d+r_ld)
        end
    end

    variables.q[1:a_size_neg_nozero,:] = Δ*q_update + (1-Δ)*q_p[1:a_size_neg_nozero,:]
end

function household_func!(
    variables::mut_var,
    parameters::NamedTuple;
    tol = tol_h,
    iter_max = iter_max
    )
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
        # price_func!(q_p, variables, parameters)

        # check convergence
        crit = max(norm(variables.V_good .- V_good_p, Inf), norm(variables.V_bad .- V_bad_p, Inf), norm(variables.q .- q_p, Inf))

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end

function density_func!(
    variables::mut_var,
    parameters::NamedTuple;
    tol = tol_μ,
    iter_max = iter_max
    )

    @unpack ψ_H = parameters
    @unpack x_size, x_Γ = parameters
    @unpack a_size, a_grid, a_size_pos, a_grid_pos = parameters
    @unpack a_size_μ, a_grid_μ, a_size_pos_μ, a_grid_pos_μ, a_ind_zero_μ = parameters

    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving invariant density: ")

    μ_good_p = similar(variables.μ_good)
    μ_bad_p = similar(variables.μ_bad)

    while crit > tol && iter < iter_max

        copyto!(μ_good_p, variables.μ_good)
        copyto!(μ_bad_p, variables.μ_bad)
        variables.μ_good .= 0.0
        variables.μ_bad .= 0.0

        # @showprogress 1 "Iterating good credit history: "
        for x_i in 1:x_size
            policy_a_good_itp = Spline1D(a_grid, variables.policy_a_good[:,x_i], k = 1)
            a_ind_default = findall(variables.policy_d_good[:,x_i] .== 1.0)[end]
            a_default = a_grid[a_ind_default]
            a_repay = a_grid[a_ind_default+1]

            for a_i in 1:a_size_μ
                a_μ = a_grid_μ[a_i]

                # remain good
                if abs(a_μ - a_repay) <= abs(a_μ - a_default)
                    a_p = clamp(policy_a_good_itp(a_grid_μ[a_i]), a_grid[1], a_grid[end])
                    ind_lower_a_p = findall(a_grid_μ .<= a_p)[end]
                    ind_upper_a_p = findall(a_p .<= a_grid_μ)[1]
                    if ind_lower_a_p != ind_upper_a_p
                        a_lower_a_p = a_grid_μ[ind_lower_a_p]
                        a_upper_a_p = a_grid_μ[ind_upper_a_p]
                        weight_lower = (a_upper_a_p - a_p) / (a_upper_a_p - a_lower_a_p)
                        weight_upper = (a_p - a_lower_a_p) / (a_upper_a_p - a_lower_a_p)
                    else
                        weight_lower = 0.5
                        weight_upper = 0.5
                    end
                    for x_p_i in 1:x_size
                        variables.μ_good[ind_lower_a_p, x_p_i] += x_Γ[x_i,x_p_i] * weight_lower * μ_good_p[a_i,x_i]
                        variables.μ_good[ind_upper_a_p, x_p_i] += x_Γ[x_i,x_p_i] * weight_upper * μ_good_p[a_i,x_i]
                    end
                # become bad
                else
                    for x_p_i in 1:x_size
                        variables.μ_bad[1, x_p_i] += x_Γ[x_i,x_p_i] * μ_good_p[a_i,x_i]
                    end
                end


            end
        end

        # @showprogress 2 "Iterating bad credit history: "
        for x_i in 1:x_size
            policy_a_bad_itp = Spline1D(a_grid_pos, variables.policy_a_bad[:,x_i], k = 1)
            for a_i in 1:a_size_pos_μ
                a_p = clamp(policy_a_bad_itp(a_grid_pos_μ[a_i]), a_grid_pos[1], a_grid_pos[end])
                ind_lower_a_p = findall(a_grid_pos_μ .<= a_p)[end]
                ind_upper_a_p = findall(a_p .<= a_grid_pos_μ)[1]
                if ind_lower_a_p != ind_upper_a_p
                    a_lower_a_p = a_grid_pos_μ[ind_lower_a_p]
                    a_upper_a_p = a_grid_pos_μ[ind_upper_a_p]
                    weight_lower = (a_upper_a_p - a_p) / (a_upper_a_p - a_lower_a_p)
                    weight_upper = (a_p - a_lower_a_p) / (a_upper_a_p - a_lower_a_p)
                else
                    weight_lower = 0.5
                    weight_upper = 0.5
                end
                for x_p_i in 1:x_size
                    # remain bad
                    variables.μ_bad[ind_lower_a_p, x_p_i] += (1-ψ_H) * x_Γ[x_i,x_p_i] * weight_lower * μ_bad_p[a_i,x_i]
                    variables.μ_bad[ind_upper_a_p, x_p_i] += (1-ψ_H) * x_Γ[x_i,x_p_i] * weight_upper * μ_bad_p[a_i,x_i]
                    # become good
                    variables.μ_good[ind_lower_a_p + a_ind_zero_μ - 1, x_p_i] += ψ_H * x_Γ[x_i,x_p_i] * weight_lower * μ_bad_p[a_i,x_i]
                    variables.μ_good[ind_upper_a_p + a_ind_zero_μ - 1, x_p_i] += ψ_H * x_Γ[x_i,x_p_i] * weight_upper * μ_bad_p[a_i,x_i]
                end
            end
        end

        μ_sum = sum(variables.μ_good) + sum(variables.μ_bad)
        variables.μ_good .= variables.μ_good ./ μ_sum
        variables.μ_bad .= variables.μ_bad./ μ_sum

        # check convergence
        crit = max(norm(variables.μ_good .- μ_good_p, Inf), norm(variables.μ_bad .- μ_bad_p, Inf))

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end

function aggregate_func!(
    variables::mut_var,
    parameters::NamedTuple
    )

    @unpack x_size, a_grid = parameters
    @unpack a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ = parameters

    variables.aggregate_var .= 0.0

    for x_i in 1:x_size
        qa_i = Spline1D(a_grid, a_grid .* variables.q[:,x_i], k = 1)
        for a_μ_i in 1:(a_size_neg_μ-1)
            a_μ = a_grid_neg_μ[a_μ_i]
            variables.aggregate_var[1] += -(variables.μ_good[a_μ_i,x_i] * qa_i(a_μ))
        end
    end
    variables.aggregate_var[2] = sum(variables.μ_good[(a_ind_zero_μ+1):end,:] .* repeat(a_grid_pos_μ[2:end],1,x_size)) + sum(variables.μ_bad[2:end,:] .* repeat(a_grid_pos_μ[2:end],1,x_size))
    variables.aggregate_var[3] = variables.aggregate_var[1] - variables.aggregate_var[2]
    variables.aggregate_var[4] = variables.aggregate_var[1] / variables.aggregate_var[2]
end

function solve_func!(
    variables::mut_var,
    parameters::NamedTuple;
    tol_h = 1E-8,
    tol_μ = 1E-8,
    iter_max = 1E+5
    )

    # solve the household's problem (including price schemes)
    household_func!(variables, parameters; tol = tol_h, iter_max = iter_max)

    # update the cross-sectional distribution
    # density_func!(variables, parameters; tol = tol_μ, iter_max = iter_max)

    # compute aggregate variables
    # aggregate_func!(variables, parameters)

    ED = variables.aggregate_var[4] - parameters.LR

    data_spec = Any[#=1=# "Multiplier"             parameters.λ;
                    #=2=# "Actual Leverage Ratio"  variables.aggregate_var[4];
                    #=3=# "Implied Leverage Ratio" parameters.LR;
                    #=4=# "Difference"             ED]

    pretty_table(data_spec, ["Name", "Value"];
                 alignment=[:l,:r],
                 formatters = ft_round(8),
                 body_hlines = [1,3])

    return ED
end

parameters = para_func()
variables = var_func(parameters)

solve_func!(variables, parameters)

plot(parameters.a_grid_neg,variables.q[1:parameters.a_ind_zero,1:parameters.e_size], seriestype=:scatter)
