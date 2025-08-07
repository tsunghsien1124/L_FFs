function para_func(;
    β::Real = 0.96,             # discount factor (households)
    σ::Real = 2.00,             # CRRA coefficient
    η::Real = 0.40,             # garnishment rate
    z::Real = 0.00,             # aggregate endowment shock
    r_f::Real = 0.03,           # risk-free saving rate
    ψ::Real = 5.00,             # upper bound of leverage ratio
    λ::Real = 0.02,             # multiplier of incentive constraint
    ω::Real = 0.20,             # capital injection rate
    e_ρ::Real = 0.90,           # AR(1) of endowment shock
    e_σ::Real = 0.15,           # s.d. of endowment shock
    e_size::Integer = 9,        # no. of endowment shock
    ν_s::Real = 0.75,           # scale of patience
    ν_p::Real = 0.25,           # probability of patience
    ν_size::Integer = 2,        # no. of preference shock
    a_min::Real = -1.00,        # min of asset holding
    a_max::Real = 50.0,         # max of asset holding
    a_size_neg::Integer = 101,  # number of grid of negative asset holding for VFI
    a_size_pos::Integer = 21,   # number of grid of positive asset holding for VFI
    a_degree::Integer = 3,      # curvature of the positive asset gridpoints
    μ_scale::Integer = 10       # scale governing the number of grids in computing density
    )
    """
    contruct an immutable object containg all paramters
    """

    # endowment shock
    e_M = tauchen(e_size, e_ρ, e_σ, 0.0, 3)
    e_Γ = e_M.p
    e_grid = collect(e_M.state_values)

    # preference schock
    ν_grid = [ν_s, 1.0]
    ν_Γ = repeat([ν_p 1.0-ν_p], ν_size, 1)

    # idiosyncratic transition matrix
    x_Γ = kron(ν_Γ, e_Γ)
    x_grid = gridmake(e_grid, ν_grid)
    x_ind = gridmake(1:e_size, 1:ν_size)
    x_size = e_size*ν_size

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length = a_size_neg))
    a_grid_pos = ((range(0.0, stop = a_size_pos-1, length = a_size_pos)/(a_size_pos-1)).^a_degree)*a_max
    a_grid = cat(a_grid_neg, a_grid_pos[2:end], dims = 1)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero,a_grid)[]

    # asset holding grid for μ
    a_size_neg_μ = convert(Int, a_size_neg)
    a_grid_neg_μ = collect(range(a_min, 0.0, length = a_size_neg_μ))
    a_size_pos_μ = convert(Int, (a_size_pos-1)*μ_scale+1)
    a_grid_pos_μ = collect(range(0.0, a_max, length = a_size_pos_μ))
    a_grid_μ = cat(a_grid_neg_μ, a_grid_pos_μ[2:end], dims = 1)
    a_size_μ = length(a_grid_μ)
    a_ind_zero_μ = findall(iszero,a_grid_μ)[]

    # return values
    return (β = β, σ = σ, η = η, z = z, r_f = r_f, ψ = ψ, λ = λ, ω = ω,
            a_degree = a_degree, μ_scale = μ_scale,
            e_ρ = e_ρ, e_σ = e_σ, e_size = e_size, e_Γ = e_Γ, e_grid = e_grid,
            ν_s = ν_s, ν_p = ν_p, ν_size = ν_size, ν_Γ = ν_Γ, ν_grid = ν_grid,
            x_Γ = x_Γ, x_grid = x_grid, x_ind = x_ind, x_size = x_size,
            a_grid = a_grid, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos,
            a_size = a_size, a_size_neg = a_size_neg, a_size_pos = a_size_pos,
            a_ind_zero = a_ind_zero,
            a_grid_μ = a_grid_μ, a_grid_neg_μ = a_grid_neg_μ, a_grid_pos_μ = a_grid_pos_μ,
            a_size_μ = a_size_μ, a_size_neg_μ = a_size_neg_μ, a_size_pos_μ = a_size_pos_μ,
            a_ind_zero_μ = a_ind_zero_μ)
end

mutable struct MutableVariables
    """
    construct a type for mutable variables
    """
    V::Array{Float64,3}
    V_nd::Array{Float64,3}
    V_d::Array{Float64,2}
    policy_a::Array{Float64,3}
    policy_d::Array{Float64,3}
    q::Array{Float64,2}
    μ::Array{Float64,3}
    aggregate_var::Array{Float64,1}
end

function var_func(
    parameters::NamedTuple;
    load_initial_values::Integer = 0
    )
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack a_size, a_size_neg, a_size_μ, e_size, ν_size, r_f, λ = parameters

    if load_initial_values == 1
        @load "09122020_initial_values.bson" V q μ

        # define value functions
        V_nd = zeros(a_size, e_size, ν_size)
        V_d = zeros(e_size, ν_size)

        # define policy functions
        policy_a = zeros(a_size, e_size, ν_size)
        policy_d = zeros(a_size, e_size, ν_size)
    else
        # define value functions
        V = zeros(a_size, e_size, ν_size)
        V_nd = zeros(a_size, e_size, ν_size)
        V_d = zeros(e_size, ν_size)

        # define policy functions
        policy_a = zeros(a_size, e_size, ν_size)
        policy_d = zeros(a_size, e_size, ν_size)

        # define pricing function
        q = ones(a_size_neg, e_size) ./ ((1.0 + r_f)*(1.0 + λ))

        # define the type distribution and its transition matrix
        μ_size = a_size_μ*e_size*ν_size
        μ = ones(a_size_μ, e_size, ν_size) ./ μ_size
    end

    # define aggregate variables
    aggregate_var = zeros(8)

    # return outputs
    variables = MutableVariables(V, V_nd, V_d,
                                 policy_a, policy_d,
                                 q, μ, aggregate_var)
    return variables
end

function u_func(c::Real, σ::Real)
    """
    compute utility of CRRA utility function with coefficient σ
    """
    if c > 0
        return σ == 1 ? log(c) : 1 / ((1-σ)*c^(σ-1))
    else
        return -10^8
        # return typemin(eltype(c))
    end
end

function value_func!(
    V_p::Array{Float64,3},
    q_p::Array{Float64,2},
    variables::MutableVariables,
    parameters::NamedTuple
    )
    """
    update value functions
    """

    @unpack β, σ, η, z, r_f = parameters
    @unpack a_size, a_grid, a_grid_neg, a_grid_pos, a_ind_zero = parameters
    @unpack x_size, x_grid, x_ind, ν_Γ, e_Γ = parameters

    Threads.@threads for x_i in 1:x_size

        e, ν = x_grid[x_i,:]
        e_i, ν_i = x_ind[x_i,:]

        # compute the next-period expected value funtion
        V_expt_p = (ν_Γ[ν_i,1]*V_p[:,:,1] + ν_Γ[ν_i,2]*V_p[:,:,2])*e_Γ[e_i,:]
        V_hat = ν*β*V_expt_p
        V_hat_itp = Akima(a_grid, V_hat)

        # compute defaulting value
        variables.V_d[e_i,ν_i] = u_func((1-η)*exp(z+e), σ) + V_hat[a_ind_zero]

        # compute non-defaulting value
        q = q_p[:,e_i]
        qa = [q.*a_grid_neg; a_grid_pos[2:end]./(1.0+r_f)]
        qa_itp = Akima(a_grid, qa)

        # Search initial value with discrete gridpoints
        qa_ind = argmin(qa)

        # set up objective function and its gradient
        object_rbl(a_p) = qa_itp(a_p[1])
        function gradient_rbl!(G, a_p)
            G[1] = derivative(object_rbl, a_p[1])
        end

        # make sure the initial value is not on the boundaries
        if a_grid[qa_ind] <= a_grid[1]
            initial = a_grid[1] + 10^(-6)
        else
            initial = a_grid[qa_ind]
        end

        # find the risky borrowing limit
        # inner_optimizer = GradientDescent(linesearch = LineSearches.BackTracking())
        inner_optimizer = GradientDescent()
        res_rbl = optimize(object_rbl, gradient_rbl!,
                           [a_grid[1]], [a_grid[a_ind_zero]],
                           [initial],
                           Fminbox(inner_optimizer))
        rbl = Optim.minimizer(res_rbl)[]

        Threads.@threads for a_i in 1:a_size
            a = a_grid[a_i]
            CoH = exp(z+e) + a

            if CoH - qa_itp(rbl) >= 0.0
                # Search initial value with discrete gridpoints
                V_all = u_func.(CoH .- qa, σ) .+ V_hat
                V_max_ind = argmax(V_all)

                # set up objective function and its gradient
                object_nd(a_p) = -(u_func(CoH - qa_itp(a_p[1]), σ) + V_hat_itp(a_p[1]))
                function gradient_nd!(G, a_p)
                    G[1] = derivative(object_nd, a_p[1])
                end

                # make sure the initial value is not on the boundaries
                if a_grid[V_max_ind] >= CoH
                    initial = CoH - 10^(-6)
                elseif a_grid[V_max_ind] <= rbl
                    initial = rbl + 10^(-6)
                else
                    initial = a_grid[V_max_ind]
                end

                # find the optimal asset holding
                res_nd = optimize(object_nd, gradient_nd!,
                                  # [rbl], [CoH],
                                  [a_grid[1]], [CoH],
                                  [initial],
                                  Fminbox(inner_optimizer))

                # record non-defaulting value funtion
                variables.V_nd[a_i,e_i,ν_i] = -Optim.minimum(res_nd)
                variables.policy_a[a_i,e_i,ν_i] = Optim.minimizer(res_nd)[]
            else
                # record non-defaulting value funtion
                variables.V_nd[a_i,e_i,ν_i] = u_func(CoH - qa_itp(rbl), σ) + V_hat_itp(rbl)
                variables.policy_a[a_i,e_i,ν_i] = rbl
            end

            # determine value function
            if variables.V_d[e_i,ν_i] > variables.V_nd[a_i,e_i,ν_i]
                variables.V[a_i,e_i,ν_i] = variables.V_d[e_i,ν_i]
                variables.policy_d[a_i,e_i,ν_i] = 1.0
            else
                variables.V[a_i,e_i,ν_i] = variables.V_nd[a_i,e_i,ν_i]
                variables.policy_d[a_i,e_i,ν_i] = 0.0
            end
        end
    end
end

function find_threshold_func(
    V_nd::Array{Float64,1},
    V_d::Array{Float64,1},
    e_grid::Array{Float64,1}
    )
    """
    compute the threshold below which households file for bankruptcy
    """

    V_diff = V_nd .- V_d
    V_nd_itp = Akima(e_grid, V_nd)
    V_d_itp = Akima(e_grid, V_d)
    V_diff_itp(e) = V_nd_itp(e) - V_d_itp(e)

    if all(V_diff .> 0.0)
        e_p_thres = -Inf
    elseif all(V_diff .< 0.0)
        e_p_thres = Inf
    else
        e_p_lower = e_grid[findall(V_diff .<= 0.0)[end]]
        e_p_upper = e_grid[findall(V_diff .>= 0.0)[1]]
        e_p_thres = find_zero(e_p->V_diff_itp(e_p), (e_p_lower, e_p_upper), Bisection())
    end
    return e_p_thres
end

function price_func!(
    q_p::Array{Float64,2},
    variables::MutableVariables,
    parameters::NamedTuple
    )
    """
    update price function
    """

    @unpack r_f, λ, η, a_grid_neg, a_size_neg, e_size, e_grid, e_ρ, e_σ, ν_p = parameters

    # parameter controling update speed
    Δ = 0.7

    # create the container
    q_update = zeros(a_size_neg, e_size)

    for a_p_i in 1:a_size_neg

        # compute defaulting threshold for (im)patient households
        e_p_thres_1 = find_threshold_func(variables.V_nd[a_p_i,:,1], variables.V_d[:,1], e_grid)
        e_p_thres_2 = find_threshold_func(variables.V_nd[a_p_i,:,2], variables.V_d[:,2], e_grid)

        if a_p_i == a_size_neg
            q_update[a_p_i,:] .= 1.0 / ((1.0+r_f)*(1.0+λ))
        else
            for e_i in 1:e_size
                e = e_grid[e_i]
                dist = Normal(0.0,1.0)

                # compute repayment rate
                default_prob_1 = cdf(dist, (e_p_thres_1-e_ρ*e)/e_σ)
                default_prob_2 = cdf(dist, (e_p_thres_2-e_ρ*e)/e_σ)
                repayment = ν_p*(1.0-default_prob_1) + (1.0-ν_p)*(1.0-default_prob_2)

                # compute garnishment
                # garnishment_1 = cdf(dist, (e_p_thres_1-(e_ρ*e+e_σ^2))/e_σ)
                # garnishment_2 = cdf(dist, (e_p_thres_2-(e_ρ*e+e_σ^2))/e_σ)
                # garnishment = (η/-a_grid_neg[a_p_i])*exp(e_ρ*e + (e_σ^2/2.0))*(ν_p*garnishment_1 + (1.0-ν_p)*garnishment_2)

                # update bond price
                # q_update[a_p_i,e_i] = clamp(repayment + garnishment, 0.0, 1.0) / ((1.0+r_f)*(1.0+λ))
                q_update[a_p_i,e_i] = clamp(repayment, 0.0, 1.0) / ((1.0+r_f)*(1.0+λ))
            end
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
    """
    update value and price functions simultaneously
    """

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving household's maximization (one loop): ")

    # initialize the next-period value functions
    V_p = similar(variables.V)
    q_p = similar(variables.q)

    while crit > tol && iter < iter_max

        # copy the current value functions to the pre-specified containers
        copyto!(V_p, variables.V)
        copyto!(q_p, variables.q)

        # update value function
        value_func!(V_p, q_p, variables, parameters)

        # update price function
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
    update the cross-sectional distribution
    """

    @unpack x_size, x_ind, e_Γ, ν_Γ, a_grid, a_size_μ, a_grid_μ, a_ind_zero_μ = parameters

    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving invariant density: ")

    # copy the previous values
    μ_p = similar(variables.μ)

    while crit > tol && iter < iter_max

        copyto!(μ_p, variables.μ)
        variables.μ .= 0.0

        for x_i in 1:x_size
            e_i, ν_i = x_ind[x_i,:]

            # interpolate decision rules
            policy_a_itp = Akima(a_grid, variables.policy_a[:,e_i,ν_i])
            policy_d_itp = Akima(a_grid, variables.policy_d[:,e_i,ν_i])

            # loop over the dimension of asset holding
            for a_i in 1:a_size_μ

                # locate it in the original grid
                a_μ = a_grid_μ[a_i]
                a_p = clamp(policy_a_itp(a_μ), a_grid[1], a_grid[end])
                ind_lower_a_p = findall(a_grid_μ .<= a_p)[end]
                ind_upper_a_p = findall(a_p .<= a_grid_μ)[1]

                # compute weights
                if ind_lower_a_p != ind_upper_a_p
                    a_lower_a_p = a_grid_μ[ind_lower_a_p]
                    a_upper_a_p = a_grid_μ[ind_upper_a_p]
                    weight_lower = (a_upper_a_p - a_p) / (a_upper_a_p - a_lower_a_p)
                    weight_upper = (a_p - a_lower_a_p) / (a_upper_a_p - a_lower_a_p)
                else
                    weight_lower = 0.5
                    weight_upper = 0.5
                end

                # loop over the dimension of exogenous individual states
                for x_p_i in 1:x_size
                    e_p_i, ν_p_i = x_ind[x_p_i,:]

                    # update the values
                    variables.μ[ind_lower_a_p,e_p_i,ν_p_i] += (1.0-policy_d_itp(a_μ)) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_lower * μ_p[a_i,e_i,ν_i]
                    variables.μ[ind_upper_a_p,e_p_i,ν_p_i] += (1.0-policy_d_itp(a_μ)) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_upper * μ_p[a_i,e_i,ν_i]
                    variables.μ[a_ind_zero_μ,e_p_i,ν_p_i] += policy_d_itp(a_μ) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * μ_p[a_i,e_i,ν_i]
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
    """
    compute aggregate variables
    """

    @unpack x_size, x_ind, e_size, e_grid, ν_size, a_grid, a_grid_neg, a_grid_pos = parameters
    @unpack a_grid_μ, a_size_μ, a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ, a_size_pos_μ = parameters
    @unpack r_f, λ, ω = parameters

    variables.aggregate_var .= 0.0

    # total loans, net worth, share of defaulters
    for x_i in 1:x_size
        # extract indices
        e_i, ν_i = x_ind[x_i,:]

        # create interpolated fuinctions
        policy_a_itp = Akima(a_grid, variables.policy_a[:,e_i,ν_i])
        policy_d_itp = Akima(a_grid, variables.policy_d[:,e_i,ν_i])
        qa = [variables.q[:,e_i].*a_grid_neg; a_grid_pos[2:end]./(1.0+r_f)]
        qa_itp = Akima(a_grid, qa)

        # loop over possible endogenous states
        for a_μ_i in 1:a_size_μ
            a_μ = a_grid_μ[a_μ_i]

            if policy_a_itp(a_μ) < 0.0
                # total loans
                variables.aggregate_var[1] += (1.0-policy_d_itp(a_μ)) * -qa_itp(policy_a_itp(a_μ)) * variables.μ[a_μ_i,e_i,ν_i]
                # debt-to-earings ratio
                variables.aggregate_var[2] += (1.0-policy_d_itp(a_μ)) * -qa_itp(policy_a_itp(a_μ)) * variables.μ[a_μ_i,e_i,ν_i] / exp(e_grid[e_i])
            else
                # total deposits
                variables.aggregate_var[3] += (1.0-policy_d_itp(a_μ)) * qa_itp(policy_a_itp(a_μ)) * variables.μ[a_μ_i,e_i,ν_i]
            end

            # net worth
            variables.aggregate_var[4] = variables.aggregate_var[1] - variables.aggregate_var[3]
            if a_μ < 0.0
                variables.aggregate_var[5] += (1.0-policy_d_itp(a_μ)) * (-a_μ) * variables.μ[a_μ_i,e_i,ν_i]
            else
                variables.aggregate_var[5] -= (1+r_f) * a_μ * variables.μ[a_μ_i,e_i,ν_i]
            end

            # capital injection ratio
            variables.aggregate_var[6] = variables.aggregate_var[4]/variables.aggregate_var[5]

            # share of defaulters
            variables.aggregate_var[7] += policy_d_itp(a_μ)*variables.μ[a_μ_i,e_i,ν_i]
        end
    end

    # leverage ratio
    variables.aggregate_var[8] = variables.aggregate_var[1] / variables.aggregate_var[4]
end

function solve_func!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol_h = 1E-8,
    tol_μ = 1E-10,
    iter_max = 500
    )

    # solve the household's problem (including price schemes)
    household_func!(variables, parameters; tol = tol_h, iter_max = iter_max)

    # update the cross-sectional distribution
    density_func!(variables, parameters; tol = tol_μ, iter_max = iter_max)

    # compute aggregate variables
    aggregate_func!(variables, parameters)

    ED = variables.aggregate_var[1] - parameters.ψ*variables.aggregate_var[4]

    data_spec = Any[#=1=# "Multiplier"            parameters.λ;
                    #=2=# "Total Loans (LHS)"     variables.aggregate_var[1];
                    #=3=# "Total Loans (RHS)"     parameters.ψ*variables.aggregate_var[4];
                    #=4=# "Difference"            ED]

    pretty_table(data_spec, ["Name", "Value"];
                 alignment=[:l,:r],
                 formatters = ft_round(12),
                 body_hlines = [1,3])

    # save results
    V = variables.V
    q = variables.q
    μ = variables.μ
    @save "09122020_initial_values.bson" V q μ

    return ED
end
