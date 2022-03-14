function value_func_RHS(
    V_p::Array{Float64,3},
    q_p::Array{Float64,2},
    parameters::NamedTuple
    )
    """
    update RHS of value functions
    """

    @unpack β_H, σ, η, ξ_bar, z, r_f, r_lp = parameters
    @unpack a_size, a_grid, a_grid_neg, a_grid_pos, a_ind_zero = parameters
    @unpack x_size, x_grid, x_ind, e_Γ, e_size, ν_Γ, ν_size = parameters

    # create containers for value and policy functions
    V = zeros(a_size, e_size, ν_size)
    V_d = zeros(e_size, ν_size)
    V_nd = zeros(a_size, e_size, ν_size)
    policy_a = zeros(a_size, e_size, ν_size)
    policy_d = zeros(a_size, e_size, ν_size)

    Threads.@threads for x_i in 1:x_size

        e, ν = x_grid[x_i,:]
        e_i, ν_i = x_ind[x_i,:]

        # compute the next-period expected value funtion
        V_expt_p = (ν_Γ[ν_i,1]*V_p[:,:,1] + ν_Γ[ν_i,2]*V_p[:,:,2])*e_Γ[e_i,:]
        V_hat = (ν*β_H)*V_expt_p
        V_hat_itp = Akima(a_grid, V_hat)

        # compute defaulting value
        V_d[e_i,ν_i] = u_func((1-η)*exp(z+e), σ) + V_hat[a_ind_zero]

        # compute non-defaulting value
        q = q_p[:,e_i]
        qa = [q.*a_grid_neg; a_grid_pos[2:end]]
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
        res_rbl = optimize(object_rbl, gradient_rbl!,
                           [a_grid[1]], [a_grid[a_ind_zero]],
                           [initial],
                           Fminbox(GradientDescent()))

        rbl = Optim.minimizer(res_rbl)[]

        Threads.@threads for a_i in 1:a_size
            a = a_grid[a_i]
            CoH = exp(z+e) + (1+r_f*(a>0))*a

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
                elseif a_grid[V_max_ind] <= a_grid[1]
                    initial = a_grid[1] + 10^(-6)
                else
                    initial = a_grid[V_max_ind]
                end

                # find the optimal asset holding
                res_nd = optimize(object_nd, gradient_nd!,
                                  [a_grid[1]], [CoH],
                                  [initial],
                                  Fminbox(GradientDescent()))

                # record non-defaulting value funtion
                V_nd[a_i,e_i,ν_i] = -Optim.minimum(res_nd)
                policy_a[a_i,e_i,ν_i] = Optim.minimizer(res_nd)[]
            else
                # record non-defaulting value funtion
                V_nd[a_i,e_i,ν_i] = u_func(CoH - qa_itp(rbl), σ) + V_hat_itp(rbl)
                policy_a[a_i,e_i,ν_i] = rbl
            end

            # compute cutoff and its associated probability
            ξ_star = clamp(V_d[e_i,ν_i] .- V_nd[a_i,e_i,ν_i], 0.0, ξ_bar)
            G_star = ξ_star / ξ_bar

            # determine value function
            V[a_i,e_i,ν_i] = -(ξ_star^2)/(2.0*ξ_bar) + G_star*V_d[e_i,ν_i] + (1.0-G_star).*V_nd[a_i,e_i,ν_i]
            policy_d[a_i,e_i,ν_i] = G_star
        end
    end
    return V, V_d, V_nd, policy_a, policy_d
end

function price_func_RHS(
    q::Array{Float64,2},
    V_nd::Array{Float64,3},
    V_d::Array{Float64,2},
    λ::Real,
    Λ::Real,
    parameters::NamedTuple
    )
    """
    update price function with one step
    """

    @unpack ξ_bar, r_f, θ, a_size_neg, e_size, e_grid, e_ρ, e_σ, ν_p = parameters

    # create the container
    q_update = zeros(a_size_neg, e_size)

    Threads.@threads for a_p_i in 1:a_size_neg

        # compute defaulting threshold for (im)patient households
        e_p_thres_zero_1   = find_threshold_func(V_nd[a_p_i,:,1], V_d[:,1], e_grid, 0.0)
        e_p_thres_ξ_bar_1 = find_threshold_func(V_nd[a_p_i,:,1], V_d[:,1], e_grid, ξ_bar)
        e_p_thres_zero_2   = find_threshold_func(V_nd[a_p_i,:,2], V_d[:,2], e_grid, 0.0)
        e_p_thres_ξ_bar_2 = find_threshold_func(V_nd[a_p_i,:,2], V_d[:,2], e_grid, ξ_bar)

        Threads.@threads for e_i in 1:e_size

            # extract endowment level and create corresaponding distribution
            e = e_grid[e_i]
            dist = Normal(e_ρ*e, e_σ)

            # compute default probability for (im)patient households
            default_prob_1 = default_prob_func(V_nd[a_p_i,:,1], V_d[:,1],
                                               e_p_thres_zero_1, e_p_thres_ξ_bar_1,
                                               e_grid, ξ_bar, dist)
            default_prob_2 = default_prob_func(V_nd[a_p_i,:,2], V_d[:,2],
                                               e_p_thres_zero_2, e_p_thres_ξ_bar_2,
                                               e_grid, ξ_bar, dist)
            default_prob = ν_p*default_prob_1 + (1.0-ν_p)*default_prob_2

            # update bond price
            repay_prob = 1.0 - default_prob
            q_update[a_p_i,e_i] = Λ*repay_prob / (Λ*(1.0+r_f)+λ*θ)
            q[a_p_i,e_i] = clamp(q_update[a_p_i,e_i], 0.0, 1.0/(1.0+r_f+(λ*θ/λ)))
        end
    end
end

function Fsys(
    X::Array{Float64,1},
    X1::Array{Float64,1},
    eta::Array{Float64,1},
    eps::Array{Float64,1},
    parameters::NamedTuple
    )
    """
    compute the LHS and RHS values of a system

    X:    Endogenous variables in the current period
    X1:   Endogenous variables in the last period
    eta:  Expectational errors
    eps:  Exogenous aggregate shocks

    X   = [V^d, V^nd, q, N, Lprime, α, Λ, λ, μ]'
    eta = [eta_V^d, eta_V^nd, eta_q, eta_α]'
    """

    #============================================#
    # (0) extract values from assigned variables #
    #============================================#

    # extract values from X and X1
    X_ind = 0

    # defaulting value function
    V_d = zeros(e_size, ν_size)
    V_d1 = zeros(e_size, ν_size)
    for x_i in 1:x_size
        X_ind += 1
        e_i, ν_i = x_ind[x_i,:]
        V_d[e_i,ν_i] = X[X_ind]
        V_d1[e_i,ν_i] = X1[X_ind]
    end

    # non-defaulting value function
    V_nd = zeros(a_size, e_size, ν_size)
    V_nd1 = zeros(a_size, e_size, ν_size)
    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        for a_i in 1:a_size
            X_ind += 1
            V_nd[a_i,e_i,ν_i] = X[X_ind]
            V_nd1[a_i,e_i,ν_i] = X1[X_ind]
        end
    end

    # price function
    q = zeros(a_size_neg, e_size)
    q1 = zeros(a_size_neg, e_size)
    for e_i in 1:e_size
        for a_i in 1:a_size_neg
            X_ind += 1
            q[a_i,e_i] = X[X_ind]
            q1[a_i,e_i] = X1[X_ind]
        end
    end

    # aggregate variables
    X_ind += 1; N = X[X_ind]; N1 = X1[X_ind]
    X_ind += 1; Lprime = X[X_ind]; Lprime1 = X1[X_ind]
    X_ind += 1; α = X[X_ind]; α1 = X1[X_ind]
    X_ind += 1; Λ = X[X_ind]; Λ1 = X1[X_ind]
    X_ind += 1; λ = X[X_ind]; λ1 = X1[X_ind]

    # distribution
    μ = zeros(a_size_μ, e_size, ν_size)
    μ1 = zeros(a_size_μ, e_size, ν_size)
    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        for a_i in 1:a_size_μ
            X_ind += 1
            μ[a_i,e_i,ν_i] = X[X_ind]
            μ1[a_i,e_i,ν_i] = X1[X_ind]
        end
    end

    # extract values from eta
    eta_ind = 0

    # defaulting value function
    eta_V_d = zeros(e_size, ν_size)
    for x_i in 1:x_size
        eta_ind += 1
        e_i, ν_i = x_ind[x_i,:]
        eta_V_d[e_i,ν_i] = eta[eta_ind]
    end

    # non-defaulting value function
    eta_V_nd = zeros(a_size, e_size, ν_size)
    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        for a_i in 1:a_size
            eta_ind += 1
            eta_V_nd[a_i,e_i,ν_i] = eta[eta_ind]
        end
    end

    # price function
    eta_q = zeros(a_size_neg, e_size)
    for e_i in 1:e_size
        for a_i in 1:a_size_neg
            eta_ind += 1
            eta_q[a_i,e_i] = eta[eta_ind]
        end
    end

    # bank's margianl benefit of net worth
    eta_ind += 1; eta_α = eta[eta_ind]

    #==============================================#
    # (1) compute implied RHS of Bellman equations #
    #==============================================#

    # create containers for value functions
    V_RHS = zeros(a_size, e_size, ν_size)
    V_d_RHS = zeros(e_size, ν_size)
    V_nd_RHS = zeros(a_size, e_size, ν_size)
    V = zeros(a_size, e_size, ν_size)

    # create containers for policy functions
    policy_d = zeros(a_size, e_size, ν_size)
    policy_a1 = zeros(a_size, e_size, ν_size)
    policy_d1 = zeros(a_size, e_size, ν_size)

    # compute continuation value
    for x_i in 1:x_size
        e_i, ν_i = x_ind[x_i,:]
        for a_i in 1:a_size
            ξ_star = clamp(V_d[e_i,ν_i] - V_nd[a_i,e_i,ν_i], 0.0, ξ_bar)
            G_star = ξ_star / ξ_bar
            policy_d[a_i,e_i,ν_i] = G_star
            V[a_i,e_i,ν_i] = -(ξ_star^2)/(2.0*ξ_bar) + G_star*V_d[e_i,ν_i] + (1.0-G_star)*V_nd[a_i,e_i,ν_i]
        end
    end

    # compute the RHS values
    V_RHS, V_nd_RHS, V_d_RHS, policy_a1, policy_d1  = value_func_onestep(V, q1, parameters)

    #==============================================#
    # (2) compute implied RHS of pricing equations #
    #==============================================#

    # create containers for pricing function
    q_RHS = zeros(a_size, e_size)

    # compute the RHS values
    price_func!(q_RHS, V_nd, V_d, λ1, Λ, parameters)

    #=========================================#
    # (2) compute implied RHS of distribution #
    #=========================================#

    #=========================================#
    # (3) compute implied aggregate variables #
    #=========================================#

end
