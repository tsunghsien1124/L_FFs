function para_func_MIT(
    parameters::NamedTuple;
    ρ_z::Real = 0.90,                       # AR(1) coefficient of aggregate shock
    σ_z::Real = 0.01,                       # s.d. of aggregate shock
    T_size::Integer = 250,                  # time periods
    time_varying_volatility::Integer = 0    # time varying volatility
    )
    """
    construct time-dependent exogenous processes
    """

    @unpack e_size, e_grid, e_ρ, e_σ = parameters

    # aggregate shock
    z_path = zeros(T_size)
    z_path[1] = -σ_z
    for T_ind in 2:T_size
        z_path[T_ind] = ρ_z*z_path[T_ind-1]
    end

    # time-varying volatility
    if time_varying_volatility == 1
        e_σ_path = zeros(T_size)
        for T_ind in 1:T_size
            e_σ_path[T_ind] = e_σ*(1-z_path[T_ind])
        end
    else
        e_σ_path = ones(T_size) .* e_σ
    end

    # Tauchen method of fixed grids
    function tauchen_Γ_func(
        size::Integer,
        grid::Array{Float64,1},
        ρ::Real,
        σ::Real
        )

        Γ = zeros(size,size)
        d = grid[2]-grid[1]
        for i in 1:size
            dist = Normal(ρ*e_grid[i],σ)
            for j in 1:size
                if j == 1
                    prob = cdf(dist, grid[j]+d/2)
                elseif j == size
                    prob = 1.0 - cdf(dist, grid[j]-d/2)
                else
                    prob = cdf(dist, grid[j]+d/2) - cdf(dist, grid[j]-d/2)
                end
                Γ[i,j] = prob
            end
        end
        return Γ
    end

    # time-varying uncertainty
    e_Γ_path = zeros(e_size, e_size, T_size)
    for T_ind in 1:T_size
        e_Γ_path[:,:,T_ind] = tauchen_Γ_func(e_size, e_grid, e_ρ, e_σ_path[T_ind])
    end

    # return values
    return (T_size = T_size, z_path = z_path, e_σ_path = e_σ_path, e_Γ_path = e_Γ_path)
end

mutable struct MutableVariables_MIT
    V::Array{Float64,4}
    V_nd::Array{Float64,4}
    V_d::Array{Float64,3}
    policy_a::Array{Float64,4}
    policy_d::Array{Float64,4}
    q::Array{Float64,3}
    μ::Array{Float64,4}
    aggregate_var::Array{Float64,2}
    λ_guess::Array{Float64,1}
end

function var_func_MIT(
    λ_optimal::Real,
    variables::MutableVariables,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple;
    load_initial_values::Integer = 1
    )
    """
    construct a mutable object containing time-varying endogenous variables
    """

    # unpack parameters
    @unpack a_size, a_size_neg, a_size_μ, e_size, ν_size = parameters
    @unpack T_size, z_path = parameters_MIT

    # define pricing related variables
    q = zeros(a_size_neg, e_size, T_size)
    q[:,:,end] = variables.q
    q[:,:,end-1] = variables.q

    # define value functions
    V = zeros(a_size, e_size, ν_size, T_size)
    V[:,:,:,end] = variables.V
    V_nd = zeros(a_size, e_size, ν_size, T_size)
    V_nd[:,:,:,end] = variables.V_nd
    V_d = zeros(e_size, ν_size, T_size)
    V_d[:,:,end] = variables.V_d

    # define policy functions
    policy_a = zeros(a_size, e_size, ν_size, T_size)
    policy_d = zeros(a_size, e_size, ν_size, T_size)

    # define the type distribution and its transition matrix
    μ_size = a_size_μ*e_size*ν_size
    μ = zeros(a_size_μ, e_size, ν_size, T_size)
    μ[:,:,:,1] = variables.μ

    # initial guess
    λ_guess = (ones(T_size) .+ z_path) .* λ_optimal

    if load_initial_values == 1
        @load "07112020_aggregate_values_MIT.bson" aggregate_var
        λ_guess[1:size(aggregate_var,2)] = aggregate_var[6,:]
    end

    # define aggregate variables (L', D', share of defaulters, N, LR', λ)
    aggregate_var = zeros(6, T_size)
    aggregate_var[1:5,end] = variables.aggregate_var
    aggregate_var[6,:] .= λ_optimal

    # return outputs
    variables_MIT = MutableVariables_MIT(V, V_nd, V_d, policy_a, policy_d, q, μ, aggregate_var, λ_guess)
    return variables_MIT
end

function value_func_MIT!(
    T_ind::Integer,
    variables_MIT::MutableVariables_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple
    )
    """
    update value functions
    """

    @unpack β, σ, η, r_f = parameters
    @unpack a_size, a_grid, a_grid_neg, a_grid_pos, a_ind_zero = parameters
    @unpack x_size, x_grid, x_ind, ν_Γ = parameters

    @unpack z_path, e_Γ_path = parameters_MIT

    z = z_path[T_ind]
    e_Γ = e_Γ_path[:,:,T_ind]

    V_p = variables_MIT.V[:,:,:,(T_ind+1)]
    q_p = variables_MIT.q[:,:,T_ind]

    Threads.@threads for x_i in 1:x_size

        e, ν = x_grid[x_i,:]
        e_i, ν_i = x_ind[x_i,:]

        # compute the next-period expected value funtion
        V_expt_p = (ν_Γ[ν_i,1]*V_p[:,:,1] + ν_Γ[ν_i,2]*V_p[:,:,2])*e_Γ[e_i,:]
        V_hat = ν*β*V_expt_p
        V_hat_itp = Akima(a_grid, V_hat)

        # compute defaulting value
        variables_MIT.V_d[e_i,ν_i,T_ind] = u_func((1-η)*exp(z+e), σ) + V_hat[a_ind_zero]

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
        inner_optimizer = GradientDescent(linesearch = LineSearches.BackTracking())
        # inner_optimizer = GradientDescent()
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
                inner_optimizer = GradientDescent(linesearch = LineSearches.BackTracking())
                # inner_optimizer = GradientDescent()
                res_nd = optimize(object_nd, gradient_nd!,
                                  [rbl], [CoH],
                                  [initial],
                                  Fminbox(inner_optimizer))

                # record non-defaulting value funtion
                variables_MIT.V_nd[a_i,e_i,ν_i,T_ind] = -Optim.minimum(res_nd)
                variables_MIT.policy_a[a_i,e_i,ν_i,T_ind] = Optim.minimizer(res_nd)[]
            else
                # record non-defaulting value funtion
                variables_MIT.V_nd[a_i,e_i,ν_i,T_ind] = u_func(CoH - qa_itp(rbl), σ) + V_hat_itp(rbl)
                variables_MIT.policy_a[a_i,e_i,ν_i,T_ind] = rbl
            end

            # determine value function
            if variables_MIT.V_d[e_i,ν_i,T_ind] > variables_MIT.V_nd[a_i,e_i,ν_i,T_ind]
                variables_MIT.V[a_i,e_i,ν_i,T_ind] = variables_MIT.V_d[e_i,ν_i,T_ind]
                variables_MIT.policy_d[a_i,e_i,ν_i,T_ind] = 1.0
            else
                variables_MIT.V[a_i,e_i,ν_i,T_ind] = variables_MIT.V_nd[a_i,e_i,ν_i,T_ind]
                variables_MIT.policy_d[a_i,e_i,ν_i,T_ind] = 0.0
            end
        end
    end
end

function price_func_MIT!(
    T_ind::Integer,
    variables_MIT::MutableVariables_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple
    )
    """
    update price function
    """

    @unpack r_f, a_size_neg, e_size, e_grid, e_ρ, e_σ, ν_p = parameters
    @unpack e_σ_path = parameters_MIT

    e_σ = e_σ_path[T_ind]
    λ = variables_MIT.λ_guess[T_ind]

    Threads.@threads for a_p_i in 1:a_size_neg

        # compute defaulting threshold for (im)patient households
        e_p_thres_zero_1 = find_threshold_func(variables_MIT.V_nd[a_p_i,:,1,T_ind+1], variables_MIT.V_d[:,1,T_ind+1], e_grid, 0.0)
        e_p_thres_zero_2 = find_threshold_func(variables_MIT.V_nd[a_p_i,:,2,T_ind+1], variables_MIT.V_d[:,2,T_ind+1], e_grid, 0.0)

        for e_i in 1:e_size
            e = e_grid[e_i]
            dist = Normal(e_ρ*e, e_σ)

            # compute default probability for (im)patient households
            default_prob_1 = cdf(dist, e_p_thres_zero_1)
            default_prob_2 = cdf(dist, e_p_thres_zero_2)
            default_prob = ν_p*default_prob_1 + (1.0-ν_p)*default_prob_2

            # update bond price
            repay_prob = clamp(1.0-default_prob, 0.0, 1.0)
            variables_MIT.q[a_p_i,e_i,T_ind] = repay_prob / ((1.0+r_f)*(1.0+λ))
        end
    end
end

function household_func_MIT!(
    variables_MIT::MutableVariables_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple,
    )

    @unpack T_size = parameters_MIT

    for T_ind in (T_size-1):(-1):1

        # update price matrices
        price_func_MIT!(T_ind, variables_MIT, parameters, parameters_MIT)

        # update value function
        value_func_MIT!(T_ind, variables_MIT, parameters, parameters_MIT)

    end
end

function density_func_MIT!(
    variables_MIT::MutableVariables_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple
    )
    """
    update the cross-sectional distribution
    """

    @unpack x_size, x_ind, ν_Γ, a_grid, a_size_μ, a_grid_μ, a_ind_zero_μ = parameters
    @unpack T_size, e_Γ_path = parameters_MIT

    variables_MIT.μ[:,:,:,2:end] .= 0.0

    for T_ind in 1:(T_size-1)

        e_Γ = e_Γ_path[:,:,T_ind]

        for x_i in 1:x_size
            e_i, ν_i = x_ind[x_i,:]

            # interpolate decision rules
            policy_a_itp = Akima(a_grid, variables_MIT.policy_a[:,e_i,ν_i,T_ind])
            policy_d_itp = Akima(a_grid, variables_MIT.policy_d[:,e_i,ν_i,T_ind])

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
                    variables_MIT.μ[ind_lower_a_p,e_p_i,ν_p_i,T_ind+1] += (1.0-policy_d_itp(a_μ)) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_lower * variables_MIT.μ[a_i,e_i,ν_i,T_ind]
                    variables_MIT.μ[ind_upper_a_p,e_p_i,ν_p_i,T_ind+1] += (1.0-policy_d_itp(a_μ)) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * weight_upper * variables_MIT.μ[a_i,e_i,ν_i,T_ind]
                    variables_MIT.μ[a_ind_zero_μ,e_p_i,ν_p_i,T_ind+1] += policy_d_itp(a_μ) * e_Γ[e_i,e_p_i] * ν_Γ[ν_i,ν_p_i] * variables_MIT.μ[a_i,e_i,ν_i,T_ind]
                end
            end
        end
        variables_MIT.μ[:,:,:,T_ind+1] .= variables_MIT.μ[:,:,:,T_ind+1] ./ sum(variables_MIT.μ[:,:,:,T_ind+1])
    end
end

function aggregate_func_MIT!(
    variables_MIT::MutableVariables_MIT,
    parameters::NamedTuple
    )
    """
    compute aggregate variables
    """

    @unpack x_size, x_ind, e_size, ν_size, a_grid, a_grid_neg, a_grid_pos = parameters
    @unpack a_grid_μ, a_size_μ, a_ind_zero_μ, a_grid_pos_μ, a_grid_neg_μ, a_size_neg_μ, a_size_pos_μ = parameters
    @unpack r_f, ψ, ω = parameters
    @unpack T_size = parameters_MIT

    variables_MIT.aggregate_var[:,1:(end-1)] .= 0.0

    for T_ind in 1:(T_size-1)

        # total loans, net worth, share of defaulters
        for x_i in 1:x_size
            # extract indices
            e_i, ν_i = x_ind[x_i,:]

            # create interpolated fuinctions
            policy_a_itp = Akima(a_grid, variables_MIT.policy_a[:,e_i,ν_i,T_ind])
            policy_d_itp = Akima(a_grid, variables_MIT.policy_d[:,e_i,ν_i,T_ind])
            qa = [variables_MIT.q[:,e_i,T_ind].*a_grid_neg; a_grid_pos[2:end]./(1.0+r_f)]
            qa_itp = Akima(a_grid, qa)

            # loop over possible endogenous states
            for a_μ_i in 1:a_size_μ
                a_μ = a_grid_μ[a_μ_i]

                # total loans and deposits
                if policy_a_itp(a_μ) < 0.0
                    variables_MIT.aggregate_var[1,T_ind] += (1.0-policy_d_itp(a_μ)) * -qa_itp(policy_a_itp(a_μ)) * variables_MIT.μ[a_μ_i,e_i,ν_i,T_ind]
                else
                    variables_MIT.aggregate_var[2,T_ind] += (1.0-policy_d_itp(a_μ)) * qa_itp(policy_a_itp(a_μ)) * variables_MIT.μ[a_μ_i,e_i,ν_i,T_ind]
                end

                # share of defaulters
                variables_MIT.aggregate_var[3,T_ind] += policy_d_itp(a_μ)*variables_MIT.μ[a_μ_i,e_i,ν_i,T_ind]

                # net worth
                if a_μ < 0.0
                    variables_MIT.aggregate_var[4,T_ind] += ω * (1.0-policy_d_itp(a_μ)) * (-a_μ) * variables_MIT.μ[a_μ_i,e_i,ν_i,T_ind]
                end
            end
        end

        # leverage ratio
        variables_MIT.aggregate_var[5,T_ind] = variables_MIT.aggregate_var[1,T_ind] / variables_MIT.aggregate_var[4,T_ind]

        # undated multiplier
        variables_MIT.aggregate_var[6,T_ind] = λ_update_func(T_ind, variables_MIT)
    end
end

function Lprime_func(
    λ_adj::Real,
    T_ind::Integer,
    variables_MIT::MutableVariables_MIT
    )

    @unpack x_size, x_ind, a_grid, a_grid_neg, a_grid_pos, a_grid_μ, a_size_μ, r_f = parameters

    λ = variables_MIT.λ_guess[T_ind]
    Lprime = 0.0

    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]

        policy_a_itp = Akima(a_grid, variables_MIT.policy_a[:,e_i,ν_i,T_ind])
        policy_d_itp = Akima(a_grid, variables_MIT.policy_d[:,e_i,ν_i,T_ind])

        q_adj = variables_MIT.q[:,e_i,T_ind].*((1+λ)/(1+λ_adj))
        qa = [q_adj.*a_grid_neg; a_grid_pos[2:end]./(1.0+r_f)]
        qa_itp = Akima(a_grid, qa)

        for a_μ_i in 1:a_size_μ
            a_μ = a_grid_μ[a_μ_i]
            if policy_a_itp(a_μ) < 0.0
                Lprime += (1.0-policy_d_itp(a_μ)) * -qa_itp(policy_a_itp(a_μ)) * variables_MIT.μ[a_μ_i,e_i,ν_i,T_ind]
            end
        end
    end
    return Lprime
end

function λ_update_func(
    T_ind::Integer,
    variables_MIT::MutableVariables_MIT
    )

    @unpack ψ = parameters

    λ = variables_MIT.λ_guess[T_ind]
    object_λ(λ_adj) = ψ*variables_MIT.aggregate_var[4,T_ind] - Lprime_func(λ_adj, T_ind, variables_MIT)

    if object_λ(0.0) > 0.0
        λ_adj = 0.0
    elseif object_λ(λ) < 0.0
        λ_adj_lower = λ
        λ_adj_upper = Inf
        λ_adj = find_zero(λ_adj->object_λ(λ_adj), (λ_adj_lower, λ_adj_upper), Bisection())
    else
        λ_adj_lower = 0.0
        λ_adj_upper = λ
        λ_adj = find_zero(λ_adj->object_λ(λ_adj), (λ_adj_lower, λ_adj_upper), Bisection())
    end
    return λ_adj
end

function solve_func_MIT!(
    variables_MIT::MutableVariables_MIT,
    parameters::NamedTuple,
    parameters_MIT::NamedTuple;
    tol = 1E-6,
    iter_max = 500
    )

    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving the model with MIT shocks: ")

    while crit > tol && iter < iter_max

        # slow updating parameter
        if crit > 1E-4
            Δ = 0.10
        else
            Δ = 0.05
        end

        # update the trajectory of leverage ratio
        variables_MIT.λ_guess = Δ*variables_MIT.aggregate_var[6,:] + (1-Δ)*variables_MIT.λ_guess

        # solve the household's problem (including price schemes)
        household_func_MIT!(variables_MIT, parameters, parameters_MIT)

        # update the cross-sectional distribution
        density_func_MIT!(variables_MIT, parameters, parameters_MIT)

        # compute aggregate variables
        aggregate_func_MIT!(variables_MIT, parameters)

        # check convergence
        crit = norm(variables_MIT.aggregate_var[6,:] .- variables_MIT.λ_guess, Inf)

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end

    # save results
    aggregate_var = variables_MIT.aggregate_var
    @save "07112020_aggregate_values_MIT.bson" aggregate_var
end
