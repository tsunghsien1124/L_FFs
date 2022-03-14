include("FLOWMath.jl")
using Main.FLOWMath: Akima, akima, interp2d
using LinearAlgebra
using ProgressMeter
using Parameters
using QuantEcon: rouwenhorst, tauchen, gridmake, MarkovChain, stationary_distributions
using Plots
using PrettyTables
using Roots
using Optim
using Calculus: derivative
using Distributions
using SparseArrays
using BSON: @save, @load
using UnicodePlots: spy
using Expectations

function para_func(;
    ψ::Real = 0.80,             # bank's survival rate
    β_H::Real = 0.96,           # discount factor (households)
    β_B::Real = 0.90,           # discount factor (banks)
    η::Real = 0.40,             # garnishment rate
    σ::Real = 2.0,              # CRRA coefficient
    z::Real = 0.0,              # aggregate uncertainty
    r_f::Real = 0.02,           # deposit rate
    λ::Real = 0.00,             # the multiplier of incentive constraint
    θ::Real = 0.20,             # the diverted fraction
    e_ρ::Real = 0.95,           # AR(1) of endowment shock
    e_σ::Real = 0.10,           # s.d. of endowment shock
    e_size::Integer = 15,       # no. of endowment shock
    ξ_bar::Real = 0.01,         # upper bound of random utility cost
    a_min::Real = -5.0,         # min of asset holding
    a_max::Real = 350.0,        # max of asset holding
    a_size_neg::Integer = 501,  # number of the grid of negative asset holding for VFI
    a_size_pos::Integer = 51,   # number of the grid of positive asset holding for VFI
    a_degree::Integer = 3,      # curvature of the positive asset gridpoints
    μ_scale::Integer = 5        # scale governing the number of grids in computing density
    )
    """
    contruct an immutable object containg all paramters
    """

    # persistent endowment shock
    e_M = tauchen(e_size, e_ρ, e_σ, 0.0, 8)
    e_Γ = e_M.p
    e_grid = collect(e_M.state_values)

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length = a_size_neg))
    a_grid_pos = ((range(0.0, stop = a_size_pos-1, length = a_size_pos)/(a_size_pos-1)).^a_degree)*a_max
    a_grid = cat(a_grid_neg, a_grid_pos[2:end], dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero,a_grid)[]

    # asset holding grid for μ
    a_size_neg_μ = convert(Int, a_size_neg*μ_scale)
    a_grid_neg_μ = collect(range(a_min, 0.0, length = a_size_neg_μ))
    a_size_pos_μ = convert(Int, a_size_pos*μ_scale)
    a_grid_pos_μ = collect(range(0.0, a_max, length = a_size_pos_μ))
    a_grid_μ = cat(a_grid_neg_μ, a_grid_pos_μ[2:end], dims = 1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(iszero,a_grid_μ)[]

    # auxiliary indicies for density
    μ_ind = gridmake(1:a_size_μ, 1:e_size)

    # solve the steady state of ω and θ to match targeted parameters
    # λ = 1 - β_H*ψ*(1+r_f) - 10^(-4)
    # λ = 1.0 - (β_B*ψ*(1+r_f))^(1/2)
    α = (β_B*(1.0-ψ)*(1.0+r_f)) / ((1.0-λ)-β_B*ψ*(1.0+r_f))
    Λ = β_B*(1.0-ψ+ψ*α)
    r_lp = λ*θ/Λ
    LR = α/θ
    AD = LR/(LR-1)
    ω = ((r_lp*(α/θ)+(1+r_f))^(-1)-ψ)/(1-ψ)

    # return values
    return (ψ = ψ, β_H = β_H, β_B = β_B, η = η, σ = σ, z = z, r_f = r_f, λ = λ,
            θ = θ, e_ρ = e_ρ, e_σ = e_σ, e_size = e_size, e_Γ = e_Γ, e_grid = e_grid,
            ξ_bar = ξ_bar, a_min = a_min, a_max = a_max,
            a_size_neg = a_size_neg, a_size_pos = a_size_pos, a_size = a_size,
            a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos, a_grid = a_grid,
            a_ind_zero = a_ind_zero,
            a_size_neg_μ = a_size_neg_μ, a_size_pos_μ = a_size_pos_μ, a_size_μ = a_size_μ,
            a_grid_neg_μ = a_grid_neg_μ, a_grid_pos_μ = a_grid_pos_μ, a_grid_μ = a_grid_μ,
            a_ind_zero_μ = a_ind_zero_μ,
            μ_ind = μ_ind, a_degree = a_degree, μ_scale = μ_scale,
            α = α, Λ = Λ, r_lp = r_lp, LR = LR, AD = AD, ω = ω)
end

mutable struct MutableVariables
    """
    construct a mutable type for endogenous variables
    """
    q::Array{Float64,2}
    V::Array{Float64,2}
    V_nd::Array{Float64,2}
    V_d::Array{Float64,1}
    policy_a::Array{Float64,2}
    policy_d::Array{Float64,2}
    μ::Array{Float64,2}
    aggregate_var::Array{Float64,1}
end

function var_func(parameters::NamedTuple)
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack a_size, a_size_neg, a_size_μ, e_size, r_f, r_lp = parameters

    # define pricing related variables
    q = ones(a_size_neg, e_size) ./ (1.0 + r_f + r_lp)

    # define value functions
    V = zeros(a_size, e_size)
    V_nd = zeros(a_size, e_size)
    V_d = zeros(e_size)

    # define policy functions
    policy_a = zeros(a_size, e_size)
    policy_d = zeros(a_size, e_size)

    # define the type distribution and its transition matrix
    μ_size = a_size_μ*e_size
    μ = ones(a_size_μ, e_size) ./ μ_size

    # define aggregate objects
    aggregate_var = zeros(5)

    # return outputs
    variables = MutableVariables(q, V, V_nd, V_d, policy_a, policy_d, μ, aggregate_var)
    return variables
end

function u_func(c::Real, σ::Real)
    """
    compute utility of CRRA function with coefficient σ
    """
    if c > 0
        return σ == 1 ? log(c) : 1 / ((1-σ)*c^(σ-1))
    else
        return -10^15
        # return typemin(eltype(c))
    end
end

function value_func!(
    V_p::Array{Float64,2},
    q_p::Array{Float64,2},
    variables::MutableVariables,
    parameters::NamedTuple
    )
    """
    update value functions
    """

    @unpack a_size, a_grid, a_grid_pos, a_grid_neg, a_ind_zero = parameters
    @unpack e_size, e_grid, e_Γ, β_H, z, σ, η, r_f, ξ_bar = parameters

    Threads.@threads for e_i in 1:e_size

        e = e_grid[e_i]

        V_hat = β_H*V_p*e_Γ[e_i,:]
        V_hat_itp = Akima(a_grid, V_hat)

        # compute defaulting value
        variables.V_d[e_i] = u_func((1-η)*exp(z+e), σ) + V_hat[a_ind_zero]

        q = q_p[:,e_i]
        qa = [q.*a_grid_neg; a_grid_pos[2:end]]
        qa_itp = Akima(a_grid, qa)

        for a_i in 1:a_size
            a = a_grid[a_i]
            CoH = exp(z+e) + (1+r_f*(a>0))*a

            # identify the optimal regions with discrete gridpoints
            V_all = u_func.(CoH .- qa, σ) .+ V_hat
            V_max_ind = argmax(V_all)

            # solve it with interpolation method
            object_nd(a_p) = -(u_func(CoH - qa_itp(a_p[1]), σ) + V_hat_itp(a_p[1]))
            function gradient_nd!(G, a_p)
                G[1] = derivative(object_nd, a_p[1])
            end

            # make sure the initial value is not on the boundaries
            if a_grid[V_max_ind] >= CoH
                initial = CoH - 10^(-6)
            elseif a_grid[V_max_ind] <= a_grid[1]
                initial = a_grid[1] + 10^(-6)
            else
                initial = a_grid[V_max_ind]
            end
            res_nd = optimize(object_nd, gradient_nd!,
                              [a_grid[1]], [CoH],
                              [initial],
                              Fminbox(GradientDescent()))

            # record non-defaulting value funtion
            variables.V_nd[a_i,e_i] = -Optim.minimum(res_nd)
            variables.policy_a[a_i,e_i] = Optim.minimizer(res_nd)[]

        end

        # compute cutoff
        ξ_star = variables.V_d[e_i] .- variables.V_nd[:,e_i]
        clamp!(ξ_star, 0.0, ξ_bar)
        G_star = ξ_star ./ ξ_bar

        # determine value function
        variables.V[:,e_i] = -(ξ_star.^2)./(2.0*ξ_bar) .+ G_star.*variables.V_d[e_i] .+ (1.0 .- G_star).*variables.V_nd[:,e_i]
        variables.policy_d[:,e_i] = G_star
    end
end

function price_func!(
    q_p::Array{Float64,2},
    variables::MutableVariables,
    parameters::NamedTuple
    )
    """
    update the price schedule
    """

    @unpack a_size_neg, e_ρ, e_σ, e_size, e_grid, r_f, r_lp, ξ_bar = parameters

    # parameter controling update speed
    Δ = 0.7

    # create the container
    q_update = zeros(a_size_neg, e_size)

    Threads.@threads for a_p_i in 1:a_size_neg

        V_diff_zero = variables.V_nd[a_p_i,:] .- variables.V_d .- 0.0
        if all(V_diff_zero .> 0.0)
            e_p_thres_zero = -Inf
        elseif all(V_diff_zero .< 0.0)
            e_p_thres_zero = Inf
        else
            e_p_lower_zero = e_grid[findall(V_diff_zero .<= 0.0)[end]]
            e_p_upper_zero = e_grid[findall(V_diff_zero .>= 0.0)[1]]
            V_diff_itp_zero = Akima(e_grid, V_diff_zero)
            e_p_thres_zero = find_zero(e_p->V_diff_itp_zero(e_p), (e_p_lower_zero, e_p_upper_zero), Bisection())
        end

        V_diff_ξ_bar = variables.V_nd[a_p_i,:] .- variables.V_d .+ ξ_bar
        if all(V_diff_ξ_bar .> 0.0)
            e_p_thres_ξ_bar = -Inf
        elseif all(V_diff_ξ_bar .< 0.0)
            e_p_thres_ξ_bar = Inf
        else
            e_p_lower_ξ_bar = e_grid[findall(V_diff_ξ_bar .<= 0.0)[end]]
            e_p_upper_ξ_bar = e_grid[findall(V_diff_ξ_bar .>= 0.0)[1]]
            V_diff_itp_ξ_bar = Akima(e_grid, V_diff_ξ_bar)
            e_p_thres_ξ_bar = find_zero(e_p->V_diff_itp_ξ_bar(e_p), (e_p_lower_ξ_bar, e_p_upper_ξ_bar), Bisection())
        end

        for e_i in 1:e_size
            e = e_grid[e_i]
            dist = Normal(e_ρ*e, e_σ)

            # file for bankruptcy certainly
            default_prob_ξ_bar = cdf(dist, e_p_thres_ξ_bar)

            # default induced by random utility costs
            if e_p_thres_zero == -Inf
                default_prob = 0.0
            elseif e_p_thres_zero == Inf
                default_prob = 1.0
            else
                e_p_size = 200
                if e_p_thres_ξ_bar == -Inf
                    e_p_thres_ξ_bar_adj = e_grid[1]
                else
                    e_p_thres_ξ_bar_adj = e_p_thres_ξ_bar
                end
                e_p_grid = collect(range(e_p_thres_ξ_bar_adj, e_p_thres_zero, length = e_p_size))
                e_p_step = e_p_grid[2] - e_p_grid[1]
                e_p_weight = [1.0; ones(e_p_size-2)*2.0; 1.0]
                ξ_star_itp = Akima(e_grid, variables.V_d .- variables.V_nd[a_p_i,:])
                ξ_grid = ξ_star_itp.(e_p_grid); clamp!(ξ_grid, 0.0, ξ_bar)
                G_grid = ξ_grid ./ ξ_bar
                e_p_prob = pdf.(dist, e_p_grid)
                default_prob_zero = (e_p_step/2)*sum(e_p_weight.*G_grid.*e_p_prob)
                default_prob = default_prob_ξ_bar + default_prob_zero
            end

            # update bond price
            repay_prob = 1.0 - default_prob
            q_update[a_p_i,e_i] = repay_prob / (1.0+r_f+r_lp)
        end
    end
    variables.q = Δ*q_update + (1-Δ)*q_p
end

function household_func!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol = tol_h,
    iter_max = iter_max
    )

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving household's maximization (one loop): ")

    # initialize the next-period value functions
    V_p = similar(variables.V)
    q_p = similar(variables.q)

    while crit > tol && iter < iter_max

        # println("iter = $iter")
        # copy the current value functions to the pre-specified containers
        copyto!(V_p, variables.V)
        copyto!(q_p, variables.q)

        # update value function
        value_func!(V_p, q_p, variables, parameters)

        # update price, its derivative, and size of bond
        price_func!(q_p, variables, parameters)

        # check convergence
        crit = max(norm(variables.V .- V_p, Inf), norm(variables.q .- q_p, Inf))

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end

function density_func!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol = tol_μ,
    iter_max = iter_max
    )
    """
    compute the stationary cross-sectional distribution
    """

    @unpack e_Γ, e_size, a_grid = parameters
    @unpack a_size_μ, a_grid_μ, a_ind_zero_μ = parameters

    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving invariant density: ")

    μ_p = similar(variables.μ)

    while crit > tol && iter < iter_max

        copyto!(μ_p, variables.μ)
        variables.μ .= 0.0

        for e_i in 1:e_size

            # interpolate the policy function
            policy_a_itp = Akima(a_grid, variables.policy_a[:,e_i])
            policy_d_itp = Akima(a_grid, variables.policy_d[:,e_i])

            # identify repaying region
            if sum(variables.policy_d[:,e_i]) > 0.0
                a_ind_repay = findall(iszero, variables.policy_d[:,e_i])[1]
                a_default = a_grid[a_ind_repay-1]
                a_repay = a_grid[a_ind_repay]
            else
                a_default = -Inf
                a_repay = a_grid[1]
            end

            # update the distribution
            for a_i in 1:a_size_μ
                a_μ = a_grid_μ[a_i]

                # repay
                if abs(a_μ - a_repay) <= abs(a_μ - a_default)

                    a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])
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

                    for e_p_i in 1:e_size
                        variables.μ[ind_lower_a_p,e_p_i] += e_Γ[e_i,e_p_i] * weight_lower * μ_p[a_i,e_i]
                        variables.μ[ind_upper_a_p,e_p_i] += e_Γ[e_i,e_p_i] * weight_upper * μ_p[a_i,e_i]
                    end

                # default
                else

                    a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])
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

                    for e_p_i in 1:e_size
                        variables.μ[ind_lower_a_p,e_p_i] += (1-policy_d_itp(a_μ)) * e_Γ[e_i,e_p_i] * weight_lower * μ_p[a_i,e_i]
                        variables.μ[ind_upper_a_p,e_p_i] += (1-policy_d_itp(a_μ)) * e_Γ[e_i,e_p_i] * weight_upper * μ_p[a_i,e_i]
                        variables.μ[a_ind_zero_μ,e_p_i] += policy_d_itp(a_μ) * e_Γ[e_i,e_p_i] * μ_p[a_i,e_i]
                    end
                end
            end
        end
        variables.μ .= variables.μ ./ sum(variables.μ)

        # check convergence
        crit = norm(variables.μ .- μ_p, Inf)

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end

function aggregate_func!(
    variables::MutableVariables,
    parameters::NamedTuple
    )

    @unpack e_size, a_grid, a_grid_pos, a_grid_neg = parameters
    @unpack a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ = parameters

    # total loans
    for e_i in 1:e_size
        q = variables.q[:,e_i]
        qa = [q.*a_grid_neg; a_grid_pos[2:end]]
        qa_itp = Akima(a_grid, qa)
        for a_μ_i in 1:(a_size_neg_μ-1)
            a_μ = a_grid_neg_μ[a_μ_i]
            variables.aggregate_var[1] += -(variables.μ[a_μ_i,e_i] * qa_itp(a_μ))
        end
    end

    # total deposits
    variables.aggregate_var[2] = sum(variables.μ[(a_ind_zero_μ+1):end,:].*repeat(a_grid_pos_μ[2:end],1,e_size))

    # net worth
    variables.aggregate_var[3] = variables.aggregate_var[1] - variables.aggregate_var[2]

    # asset to debt ratio (or loan to deposit ratio)
    variables.aggregate_var[4] = variables.aggregate_var[1] / variables.aggregate_var[2]

    # share of defaulters
    for e_i in 1:e_size
        policy_d_itp = Akima(a_grid, variables.policy_d[:,e_i])
        for a_μ_i in 1:a_size_neg_μ
            a_μ = a_grid_neg_μ[a_μ_i]
            variables.aggregate_var[5] += (variables.μ[a_μ_i,e_i] * policy_d_itp(a_μ))
        end
    end
end

function solve_func!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol_h = 1E-8,
    tol_μ = 1E-10,
    iter_max = 5000
    )

    # solve the household's problem (including price schemes)
    household_func!(variables, parameters; tol = tol_h, iter_max = iter_max)

    # update the cross-sectional distribution
    density_func!(variables, parameters; tol = tol_μ, iter_max = iter_max)

    # compute aggregate variables
    aggregate_func!(variables, parameters)

    ED = variables.aggregate_var[4] - parameters.AD

    data_spec = Any[#=1=# "Multiplier"                   parameters.λ;
                    #=2=# "Asset-to-Debt Ratio (Demand)" variables.aggregate_var[4];
                    #=3=# "Asset-to-Debt Ratio (Supply)" parameters.AD;
                    #=4=# "Difference"                   ED]

    pretty_table(data_spec, ["Name", "Value"];
                 alignment=[:l,:r],
                 formatters = ft_round(12),
                 body_hlines = [1,3])

    # save results
    V = variables.V
    q = variables.q
    μ = variables.μ
    @save "24092020_initial_values.bson" V q μ

    return ED
end

parameters = para_func(; λ = 0.00)
variables = var_func(parameters)

println("Solving the model with $(Threads.nthreads()) threads in Julia...")
solve_func!(variables, parameters)

data_spec = Any[#= 1=# "Number of Endowment"                parameters.e_size;
                #= 2=# "Number of Assets"                   parameters.a_size;
                #= 3=# "Number of Negative Assets"          parameters.a_size_neg;
                #= 4=# "Number of Positive Assets"          parameters.a_size_pos;
                #= 5=# "Number of Assets (for Density)"     parameters.a_size_μ;
                #= 6=# "Minimum of Assets"                  parameters.a_grid[1];
                #= 7=# "Maximum of Assets"                  parameters.a_grid[end];
                #= 8=# "Exogenous Risk-free Rate"           parameters.r_f;
                #= 9=# "Multiplier of Incentive Constraint" parameters.λ;
                #=10=# "Marginal Benifit of Net Worth"      parameters.α;
                #=11=# "Diverting Fraction"                 parameters.θ;
                #=12=# "Asset-to-Debt Ratio (Supply)"       parameters.AD;
                #=13=# "Additional Opportunity Cost"        parameters.r_lp;
                #=14=# "Total Loans"                        variables.aggregate_var[1];
                #=15=# "Total Deposits"                     variables.aggregate_var[2];
                #=16=# "Net Worth"                          variables.aggregate_var[3];
                #=17=# "Asset-to-Debt Ratio (Demand)"       variables.aggregate_var[4];
                #=18=# "Share of Defaulters"                variables.aggregate_var[5]]

hl_LR = Highlighter(f      = (data,i,j) -> i == 14 || i == 19,
                    crayon = Crayon(background = :light_blue))

pretty_table(data_spec, ["Name", "Value"];
             alignment=[:l,:r],
             formatters = ft_round(4),
             body_hlines = [7,9,13],
             highlighters = hl_LR)

#=
parameters = para_func()
para_targeted(x) = para_func(; λ = x)
solve_targeted(x) = solve_func!(var_func(para_targeted(x)), para_targeted(x))
λ_lower, λ_upper = 0, 1-(parameters.β_B*parameters.ψ*(1+parameters.r_f))^(1/2)
# λ_optimal = find_zero(solve_targeted, (λ_lower, λ_upper), Bisection())
λ_optimal = find_zero(solve_targeted, (0.04987767453057565, 0.05), Bisection())
=#

#=
λ_optimal = 0.04988090645870582
parameters = para_func(; λ = λ_optimal)
variables = var_func(parameters)
println("Solving the model with $(Threads.nthreads()) threads in Julia...")
solve_func!(variables, parameters)

Vss = variables.V
V_repayss = variables.V_repay
V_defautlss = variables.V_default
qss = variables.q
μss = variables.μ
Lss = variables.aggregate_var[1]
Dss = variables.aggregate_var[2]
Nss = variables.aggregate_var[3]
λss = parameters.λ
αss = parameters.α
Λss = parameters.Λ

@save "optimal_values.bson" Vss V_repayss V_defautlss qss μss Lss Dss Nss αss λss Λss
=#

#=========================================================#
# Comparison between with and without financial frictions #
#=========================================================#
#=
λ_optimal = 0.04988090645870582
parameters_FF = para_func(; λ = λ_optimal, e_σ = 0.10)
variables_FF = var_func(parameters_FF)
solve_func!(variables_FF, parameters_FF)

parameters_NFF = para_func(; λ = 0.0, θ = 0.0, e_σ = 0.10)
variables_NFF = var_func(parameters_NFF)
solve_func!(variables_NFF, parameters_NFF)

results_compare_FF = zeros(6,3)
results_compare_FF[1:4,1] = variables_NFF.aggregate_var
results_compare_FF[5,1] = 0.0
results_compare_FF[6,1] = sum(variables_NFF.μ.*variables_NFF.policy_d_matrix)*100
results_compare_FF[1:3,2] = variables_FF.aggregate_var[1:3]
results_compare_FF[4,2] = variables_FF.aggregate_var[1]/variables_FF.aggregate_var[3]
results_compare_FF[5,2] = parameters_FF.r_lp
results_compare_FF[6,2] = sum(variables_FF.μ.*variables_FF.policy_d_matrix)*100
results_compare_FF[:,3] = ((results_compare_FF[:,2] .- results_compare_FF[:,1])./results_compare_FF[:,1])*100
=#
#=
PD_FF = parameters_FF.ν_p*variables_FF.prob_default[1:parameters_FF.a_ind_zero,:,1] + (1-parameters_FF.ν_p)*variables_FF.prob_default[1:parameters_FF.a_ind_zero,:,2]
PD_NFF = parameters_NFF.ν_p*variables_NFF.prob_default[1:parameters_NFF.a_ind_zero,:,1] + (1-parameters_NFF.ν_p)*variables_NFF.prob_default[1:parameters_NFF.a_ind_zero,:,2]

step = 10^(-2)
e_itp_FF = parameters_FF.e_grid[1]:step:parameters_FF.e_grid[end]
a_p_itp_FF = parameters_FF.a_grid_neg[1]:step:parameters_FF.a_grid_neg[end]
PD_FF_itp = interp2d(akima, parameters_FF.a_grid_neg, parameters_FF.e_grid, PD_FF, a_p_itp_FF, e_itp_FF)

plot_PD_FF = plot(e_itp_FF, a_p_itp_FF, PD_FF_itp,
                  st = :heatmap, clim = (0,1), color = :dense,
                  xlabel = "\$p\$", ylabel = "\$a'\$",
                  colorbar_title = "\$P(d'=1)\$")

e_itp_NFF = parameters_NFF.e_grid[1]:step:parameters_NFF.e_grid[end]
a_p_itp_NFF = parameters_NFF.a_grid_neg[1]:step:parameters_NFF.a_grid_neg[end]
PD_NFF_itp = interp2d(akima, parameters_NFF.a_grid_neg, parameters_NFF.e_grid, PD_NFF, a_p_itp_NFF, e_itp_NFF)

plot_PD_NFF = plot(e_itp_NFF, a_p_itp_NFF, PD_NFF_itp,
                   st = :heatmap, clim = (0,1), color = :dense,
                   xlabel = "\$p\$", ylabel = "\$a'\$",
                   colorbar_title = "\$P(d'=1)\$")

plot(plot_PD_FF, plot_PD_NFF, layout = (2,1))
savefig("plot_PD_FF.pdf")
=#

#========================================#
# Comparison of time-varying uncertainty #
#========================================#
#=
parameters = para_func()
para_targeted(x) = para_func(; λ = x, e_σ = 0.10*1.02)
solve_targeted(x) = solve_func!(var_func(para_targeted(x)), para_targeted(x))
λ_lower, λ_upper = 0, 1-(parameters.β_B*parameters.ψ*(1+parameters.r_f))^(1/2)
# λ_optimal_u = find_zero(solve_targeted, (λ_lower, λ_upper), Bisection())
λ_optimal_u = find_zero(solve_targeted, (0.026556104511, 0.026556104541), Bisection())
=#

#=
λ_optimal_u = 0.026556104511
parameters_FF_u = para_func(; λ = λ_optimal_u, e_σ = 0.10*1.02)
variables_FF_u = var_func(parameters_FF_u)
solve_func!(variables_FF_u, parameters_FF_u)

results_compare_u = zeros(6,3)
results_compare_u[1:3,1] = variables_FF.aggregate_var[1:3]
results_compare_u[4,1] = variables_FF.aggregate_var[1] / variables_FF.aggregate_var[3]
results_compare_u[5,1] = parameters_FF.r_lp
results_compare_u[6,1] = sum(variables_FF.μ.*variables_FF.policy_d_matrix)*100

results_compare_u[1:3,2] = variables_FF_u.aggregate_var[1:3]
results_compare_u[4,2] = variables_FF_u.aggregate_var[1]/variables_FF_u.aggregate_var[3]
results_compare_u[5,2] = parameters_FF_u.r_lp
results_compare_u[6,2] = sum(variables_FF_u.μ.*variables_FF_u.policy_d_matrix)*100

results_compare_u[:,3] = ((results_compare_u[:,2] .- results_compare_u[:,1])./results_compare_u[:,1])*100
=#

#=
#=======================================#
# Comparison of time-varying preference #
#=======================================#
parameters = para_func()
para_targeted(x) = para_func(; λ = x, ν_p = 0.10*1.02)
solve_targeted(x) = solve_func!(var_func(para_targeted(x)), para_targeted(x))
λ_lower, λ_upper = 0, 1-(parameters.β_B*parameters.ψ*(1+parameters.r_f))^(1/2)
# λ_optimal_u = find_zero(solve_targeted, (λ_lower, λ_upper), Bisection())
λ_optimal_ν = find_zero(solve_targeted, (0.04988090645870582, 0.06), Bisection())

λ_optimal_ν = 0.05916473242601564
parameters_FF_ν = para_func(; λ = λ_optimal_ν, ν_p = 0.10*1.02)
variables_FF_ν = var_func(parameters_FF_ν)
solve_func!(variables_FF_ν, parameters_FF_ν)

results_compare_ν = zeros(6,3)
results_compare_ν[1:3,1] = variables_FF.aggregate_var[1:3]
results_compare_ν[4,1] = variables_FF.aggregate_var[1] / variables_FF.aggregate_var[3]
results_compare_ν[5,1] = parameters_FF.r_lp
results_compare_ν[6,1] = sum(variables_FF.μ.*variables_FF.policy_d_matrix)*100

results_compare_ν[1:3,2] = variables_FF_ν.aggregate_var[1:3]
results_compare_ν[4,2] = variables_FF_ν.aggregate_var[1]/variables_FF_ν.aggregate_var[3]
results_compare_ν[5,2] = parameters_FF_ν.r_lp
results_compare_ν[6,2] = sum(variables_FF_ν.μ.*variables_FF_ν.policy_d_matrix)*100

results_compare_ν[:,3] = ((results_compare_ν[:,2] .- results_compare_ν[:,1])./results_compare_ν[:,1])*100
=#

#=
parameters_NFF_u = para_func(; λ = 0.0, θ = 0.0, e_σ = 0.10*1.02)
variables_NFF_u = var_func(parameters_NFF_u)
solve_func!(variables_NFF_u, parameters_NFF_u)
=#

#=
plot(parameters.a_grid_neg, variables.q[1:parameters.a_ind_zero,:,1], seriestype=:scatter, legend=:bottomright)
plot(parameters.a_grid_neg, repeat(parameters.a_grid_neg,1,parameters.e_size).*variables.q[1:parameters.a_ind_zero,:,1], seriestype=:scatter, legend=:bottomright)

function plot_price_itp(
    a_size_neg_itp::Integer,
    variables::mut_var,
    parameters::NamedTuple)

    @unpack e_size, a_grid, a_grid = parameters

    q_all = zeros(a_size_neg_itp, e_size)
    a_grid_itp = collect(range(a_grid[1], 0, length = a_size_neg_itp))

    for e_i in 1:e_size
        q_itp = interpolate(a_grid, variables.q[:,e_i])
        q_all[:, e_i] .= [q_itp(a_grid_itp[a_i]) for a_i in 1:a_size_neg_itp]
    end

    return q_all, a_grid_itp
end
q_all, a_grid_itp = plot_price_itp(5000, variables, parameters)
plot(a_grid_itp, q_all)
plot(a_grid_itp, q_all .* repeat(a_grid_itp,1,parameters.e_size))
=#

#=
β_B = 0.90
r_f = 0.02
ψ = 0.8
θ = 0.20
L_λ = 0
U_λ = 1 - β_B*ψ*(1+r_f)
Δ_λ = 10^(-2)
λ_grid = collect(L_λ:Δ_λ:U_λ)
α(λ) = (β_B*(1-ψ)*(1+r_f)) / ((1-λ)-β_B*ψ*(1+r_f))
Λ(λ) = β_B*(1-ψ+ψ*α(λ))
C(λ) = λ*θ / Λ(λ)
λ_optimal = 1-√(β_B*ψ*(1+r_f))
L_α = α(L_λ)
L_Λ = Λ(L_λ)
U_α = α(λ_optimal)
U_Λ = Λ(λ_optimal)
L_C = 0
U_C = C(λ_optimal)
α_grid = α.(λ_grid)
Λ_grid = Λ.(λ_grid)
C_grid = C.(λ_grid)

table = round.([L_λ λ_optimal; L_α U_α; L_Λ U_Λ; L_C U_C; L_α/θ U_α/θ],digits=4)


plot(λ_grid, [α_grid Λ_grid], seriestype=:scatter)
plot(λ_grid, C_grid, lw = 2, label = "C(λ)")
scatter!([λ_optimal], [U_C], label = "optimal λ")
savefig("plot_C_lambda.pdf")
=#

#=
PD = 1 .- variables.q[1:parameters.a_size_neg,1:parameters.p_size,1] .* (1+parameters.r_f+parameters.r_bf)
plot(parameters.e_grid, parameters.a_grid_neg, variables.prob_default[1:parameters.a_ind_zero,:,1],
     st = :heatmap, clim = (0,1), color = :dense,
     xlabel = "\$e\$", ylabel = "\$a'\$",
     xlims = (-1,0), ylims = (-1.0,0.0),
     yticks = parameters.a_grid_neg[1]:0.5:parameters.a_grid_neg[end],
     colorbar_title = "\$P(g'_h(a',s)=1)\$",
     theme = theme(:ggplot2))
=#
