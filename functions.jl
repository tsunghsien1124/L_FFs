function para(; λ::Real = 0.10,         # history rased probability
                β::Real = 0.96,         # discount factor
                ξ::Real = 0.25,         # garnishment rate
                σ::Real = 3,            # CRRA coefficient
                r::Real = 0.03,         # risk-free rate
                ρ_p::Real = 0.90,       # AR(1) of persistent shock
                σ_p::Real = 0.20,       # s.d. of persistent shock
                σ_t::Real = 0.30,       # s.d. of temporary shock
                p_size::Integer = 3,    # no. of persistent shock
                t_size::Integer = 3,    # no. of temporary shock
                e_size::Integer = 3,    # no. of expenditure shock
                a_min::Real = -1,       # min of asset holding
                a_max::Real = 10,       # max of asset holding
                a_scale::Integer = 8,   # scale of the grid asset holding
                a_degree::Integer = 3)  # degree of the grid asset holding

      # persistent shock
      Mp = rouwenhorst(p_size, ρ_p, σ_p)
      Pp = Mp.p
      p_grid = collect(Mp.state_values) .+ 1.0

      # temporary shock
      Mt = rouwenhorst(t_size, 0.0, σ_t)
      Pt = Mt.p
      t_grid = collect(Mt.state_values) .+ 1.0

      # expenditure schock
      e_grid = [0, minimum(p_grid.*t_grid)*0.4, minimum(p_grid.*t_grid)*0.9]
      Pe = repeat([0.7 0.2 0.1],e_size,1)

      # idiosyncratic transition matrix conditional
      Px = kron(Pe, kron(Pt, Pp))
      x_grid = gridmake(p_grid, t_grid, e_grid)
      x_ind = gridmake(1:p_size, 1:t_size, 1:e_size)
      x_size = p_size*t_size*e_size

      # asset holding grid
      a_size = (a_max-a_min)*a_scale + 1
      # a_grid = collect(range(a_min, stop = a_max, length = a_size))
      a_grid = ((range(0, stop = a_size-1, length = a_size)/(a_size-1)).^a_degree)*(a_max-a_min) .+ a_min

      # find the index where a = 0
      ind_a_zero = findall(a_grid .>= 0)[1]
      a_grid[ind_a_zero] = 0

      # define the size of positive asset
      a_size_neg = ind_a_zero - 1
      a_size_pos = a_size - a_size_neg

      # define the negative or positive asset holding grid
      a_grid_neg = a_grid[1:a_size_neg]
      a_grid_pos = a_grid[ind_a_zero:end]

      # return values
      return (λ = λ, β = β, ξ = ξ, σ = σ, r = r, a_grid = a_grid, ind_a_zero = ind_a_zero, a_size = a_size, a_size_pos = a_size_pos, a_size_neg = a_size_neg, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos, Pp = Pp, p_grid = p_grid, p_size = p_size, Pt = Pt, t_grid = t_grid, t_size = t_size, Pe = Pe, e_grid = e_grid, e_size = e_size, Px = Px, x_grid = x_grid, x_size = x_size, x_ind = x_ind)
end

mutable struct mut_vars
    V_bad::Array{Float64,2}
    V_good::Array{Float64,2}
    V_good_default::Array{Float64,2}
    V_good_repay::Array{Float64,2}
    policy_a_bad::Array{Float64,2}
    policy_a_good::Array{Float64,2}
    policy_a_good_default::Array{Float64,2}
    policy_a_good_repay::Array{Float64,2}
    policy_matrix_a_bad::SparseMatrixCSC{Float64,Int64}
    policy_matrix_a_good_default::SparseMatrixCSC{Float64,Int64}
    policy_matrix_a_good_repay::SparseMatrixCSC{Float64,Int64}
    transition_matrix::SparseMatrixCSC{Float64,Int64}
    q::Array{Float64,2}
    μ::Array{Float64,3}
    L::Real
    D::Real
end

function vars(parameters::NamedTuple)

    # unpack parameters
    @unpack β, ξ, σ, r, a_grid, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos, ind_a_zero, x_grid, x_size = parameters

    # define value functions
    # V_bad = zeros(a_size_pos, x_size)
    V_bad = u_func.(repeat(transpose(x_grid[:,1] .* x_grid[:,2] .- x_grid[:,3]),a_size_pos,1) .+ repeat(a_grid_pos*r,1,x_size), σ) ./ (1-β)

    # V_good_default = zeros(a_size, x_size)
    V_good_default = u_func.(repeat(transpose((1-ξ)*x_grid[:,1].*x_grid[:,2]),a_size,1),σ) ./ (1-β)

    # V_good_repay = zeros(a_size, x_size)
    V_good_repay = u_func.(repeat(transpose(x_grid[:,1] .* x_grid[:,2] .- x_grid[:,3]),a_size,1).+cat(repeat(a_grid_neg*r,1,x_size),repeat(a_grid_pos*r,1,x_size),dims=1), σ) ./ (1-β)

    # V_good = zeros(a_size, x_size)
    V_good = zeros(a_size, x_size)
    for x_i in 1:x_size, a_i in 1:a_size
        if V_good_default[a_i,x_i] >= V_good_repay[a_i,x_i]
            V_good[a_i,x_i] = V_good_default[a_i,x_i]
        else
            V_good[a_i,x_i] = V_good_repay[a_i,x_i]
        end
    end

    # define policy functions
    policy_a_bad = zeros(a_size_pos, x_size)
    policy_a_good = zeros(a_size, x_size)
    policy_a_good_default = repeat([ind_a_zero], a_size, x_size)
    policy_a_good_repay = zeros(a_size, x_size)

    # define policy matrices
    policy_matrix_a_bad = spzeros(a_size, a_size*x_size)
    policy_matrix_a_good_default = spzeros(a_size, a_size*x_size)
    policy_matrix_a_good_repay = spzeros(a_size, a_size*x_size)

    # define the transition matrix for the cross-sectional distribution
    G_size = (a_size + a_size_pos) * x_size
    transition_matrix = spzeros(G_size, G_size)

    # initialize aggregate objects
    L = 0.0
    D = 0.0

    # define pricing function and default probability
    q = ones(a_size, x_size)
    q[1:a_size_neg,:] .= 1 / (1 + r)

    # define stationary distribution
    μ = zeros(a_size, x_size, 2)
    μ[:,:,1] .= 1 / ( (a_size_pos+a_size+1) * x_size )
    μ[a_size_neg:end,:,2] .= 1 / ( (a_size_pos+a_size+1) * x_size )

    variables = mut_vars(V_bad, V_good, V_good_default, V_good_repay, policy_a_bad, policy_a_good, policy_a_good_default, policy_a_good_repay, policy_matrix_a_bad, policy_matrix_a_good_default, policy_matrix_a_good_repay, transition_matrix, q, μ, L, D)

    return variables
end

function u_func(c::Real, σ::Real)
    # compute utility
    if c > 0
        if σ == 1
            return log(c)
        else
            return 1 / ((1-σ)*c^(σ-1))
        end
    else
        # println("WARNING: non-positive consumption")
        return -Inf
    end
end

function du_func(c::Real, σ::Real)
    # compute marginal utility
    if c > 0
        return c^(-σ)
    else
        # println("WARNING: non-positive consumption!")
    end
end

function inv_du_func(x::Real, σ::Real)
    # compute marginal utility
    return x^(-1/σ)
end

function rbl_func(q_i::Array{Float64,1}, a_grid_neg::Array{Float64,1})
    # compute the risky borrowing limit for curren type (x_i)
    a_size_neg = length(a_grid_neg)
    q_func = LinearInterpolation(a_grid_neg, q_i[1:a_size_neg], extrapolation_bc = Line())
    obj_rbl(ap) = ap*q_func(ap)
    results = optimize(obj_rbl, a_grid_neg[1], 0)
    rbl = results.minimizer
    rbl_ind = minimum(findall(a_grid_neg .>= rbl))
    return rbl, rbl_ind
end

function V_hat_func(ap_i::Integer, V_p::Array{Float64,2}, β::Real, Px_i::Array{Float64,1})
    # compute the discounted expected value function for asset holding in the next period (ap_i) and current type (x_i)
    return β*sum(Px_i .* V_p[ap_i,:])
end

function dV_hat_func(ap_i::Integer, V_p::Array{Float64,2}, a_grid::Array{Float64,1}, β::Real, Px_i::Array{Float64,1})
    # compute first-order derivative of the discounted expected value function for asset holding in the next period (ap_i) and current type (x_i) wrt the first argument (ap_i) by forward finite difference
    if ap_i < size(V_p,1)
        return (V_hat_func(ap_i+1,V_p,β,Px_i) - V_hat_func(ap_i,V_p,β,Px_i)) / (a_grid[ap_i+1] - a_grid[ap_i])
    else
        return (V_hat_func(ap_i,V_p,β,Px_i) - V_hat_func(ap_i-1,V_p,β,Px_i)) / (a_grid[ap_i] - a_grid[ap_i-1])
    end
end

function ncr_func(V_p::Array{Float64,2}, a_grid::Array{Float64,1}, β::Real, Px_i::Array{Float64,1})
    # compute the non-concave region for type (x_i)
    a_size = length(a_grid)
    ncr_l, ncr_u = 1, a_size

    # (1) find the lower bound
    dV_hat_vec = dV_hat_func.(1:a_size, Ref(V_p), Ref(a_grid), β, Ref(Px_i))
    V_max, i_max = dV_hat_vec[1], 1
    while i_max < a_size
        if V_max > maximum(dV_hat_vec[(i_max+1):end])
            i_max += 1
            V_max = dV_hat_vec[i_max]
        else
            ncr_l = i_max
            break
        end
    end

    # (2) find the upper bound
    V_min, i_min = dV_hat_vec[end], a_size
    while i_min > 1
        if V_min < minimum(dV_hat_vec[1:(i_min-1)])
            i_min -= 1
            V_min = dV_hat_vec[i_min]
        else
            ncr_u = i_min
            break
        end
    end
    return ncr_l, ncr_u
end

function CoH_G_func(ap_i::Integer, V_good_p::Array{Float64,2}, a_grid::Array{Float64,1}, q_i::Array{Float64,1}, β::Real, Px_i::Array{Float64,1}, σ::Real)
    # compute the cash on hands for the case of asset holding in the next period (ap_i), current type (x_i), and good credit history with repayment. Note that cash on hands for the case of good credit history with defaulting is trivially determined so it can be ignored
    return inv_du_func(dV_hat_func(ap_i, V_good_p, a_grid, β, Px_i)/q_i[ap_i], σ) + q_i[ap_i]*a_grid[ap_i]
end

function CoH_B_func(ap_i::Integer, V_good_p_pos::Array{Float64,2}, V_bad_p::Array{Float64,2}, a_grid_pos::Array{Float64,1}, β::Real, Px_i::Array{Float64,1}, λ::Real, σ::Real)
    # compute the cash on hands for asset holding in the next period (ap_i), current type (x_i), and bad credit hostory. Note that ap must be positive (saving only)
    return inv_du_func(dV_hat_func(ap_i, λ*V_good_p_pos+(1-λ)*V_bad_p, a_grid_pos, β, Px_i),σ) + a_grid_pos[ap_i]
end

function sols_G_func(ncr_l_good::Integer, ncr_u_good::Integer, V_good_p_rbl::Array{Float64,2}, a_grid_rbl::Array{Float64,1}, q_i_rbl::Array{Float64,1}, β::Real, Px_i::Array{Float64,1}, σ::Real, r::Real, earnings::Real)
    # (1) create a matrix of local solutions for good credit history with repayment (bounded below by the risky borrowing limit), including (a) next-period asset holdings, (b) cash on hands, (c) associated value functions, (d) associated current asset holdings, and (e) identifier of a global solution

    # comnpute the size of grids above the risky borrowing limit
    a_size_rbl = length(a_grid_rbl)

    # construct the matrix storing all possible local solutions
    local_sols_G = zeros(a_size_rbl,5)
    local_sols_G[:,1] = a_grid_rbl
    local_sols_G[:,2] = CoH_G_func.(1:a_size_rbl, Ref(V_good_p_rbl), Ref(a_grid_rbl), Ref(q_i_rbl), β, Ref(Px_i), σ)
    local_sols_G[:,3] = u_func.(local_sols_G[:,2] .- q_i_rbl .* local_sols_G[:,1], σ) .+ V_hat_func.(1:a_size_rbl, Ref(V_good_p_rbl), β, Ref(Px_i))

    # (2) identify global solutions by introduing an additional discretized VFI maximization step for the points in the non-cave region

    # define the variables whose indices are within the non-concave region
    a_grid_ncr = a_grid_rbl[ncr_l_good:ncr_u_good]
    a_size_ncr = length(a_grid_ncr)
    V_good_p_ncr = V_good_p_rbl[ncr_l_good:ncr_u_good,:]
    q_i_ncr = q_i_rbl[ncr_l_good:ncr_u_good]

    # construct the matrix storing global solutions
    for ap_i in 1:a_size_rbl
        if ap_i < ncr_l_good || ap_i > ncr_u_good
            if local_sols_G[ap_i,2] - earnings >= 0
                local_sols_G[ap_i,4] = (local_sols_G[ap_i,2] - earnings)/(1+r)
                local_sols_G[ap_i,5] = 1
            else
                local_sols_G[ap_i,4] = local_sols_G[ap_i,2] - earnings
                local_sols_G[ap_i,5] = 1
            end
        else
            temp_vec = u_func.(local_sols_G[ap_i,2] .- q_i_ncr .* a_grid_ncr, σ) .+ V_hat_func.(1:a_size_ncr, Ref(V_good_p_ncr), β, Ref(Px_i))
            if (ap_i - ncr_l_good + 1) == findall(temp_vec .== maximum(temp_vec))[1]
                if local_sols_G[ap_i,2] - earnings >= 0
                    local_sols_G[ap_i,4] = (local_sols_G[ap_i,2] - earnings)/(1+r)
                    local_sols_G[ap_i,5] = 1
                else
                    local_sols_G[ap_i,4] = local_sols_G[ap_i,2] - earnings
                    local_sols_G[ap_i,5] = 1
                end
            end
        end
    end

    # return global solutions
    return local_sols_G[local_sols_G[:,5] .== 1.0, 1:4]
end

function sols_B_func(ncr_l_bad::Integer, ncr_u_bad::Integer, V_good_p_pos::Array{Float64,2}, V_bad_p::Array{Float64,2}, a_grid_pos::Array{Float64,1}, β::Real, Px_i::Array{Float64,1}, λ::Real, σ::Real, r::Real, earnings::Real)
    # (1) create a matrix of local solutions for bad credit history, including (a) next-period asset holdings, (b) cash on hands, (c) associated value functions, (d) associated current asset holdings, and (e) identifier of a global solution

    # construct the matrix storing all possible local solutions
    a_size_pos = length(a_grid_pos)
    local_sols_B = zeros(a_size_pos,5)
    local_sols_B[:,1] = a_grid_pos
    local_sols_B[:,2] = CoH_B_func.(1:a_size_pos, Ref(V_good_p_pos), Ref(V_bad_p), Ref(a_grid_pos), β, Ref(Px_i), λ, σ)
    local_sols_B[:,3] = u_func.(local_sols_B[:,2] .- local_sols_B[:,1],σ) .+ V_hat_func.(1:a_size_pos, Ref(λ*V_good_p_pos+(1-λ)*V_bad_p), β, Ref(Px_i))

    # (2) identify global solutions by introduing an additional discretized VFI maximization step for the points in the non-cave region

    # define the variables whose indices are within the non-concave region
    a_grid_pos_ncr = a_grid_pos[ncr_l_bad:ncr_u_bad]
    a_size_pos_ncr = length(a_grid_pos_ncr)
    V_good_p_pos_ncr = V_good_p_pos[ncr_l_bad:ncr_u_bad,:]
    V_bad_p_ncr = V_bad_p[ncr_l_bad:ncr_u_bad,:]

    # construct the matrix storing global solutions
    for ap_i in 1:length(a_grid_pos)
        if ap_i < ncr_l_bad || ap_i > ncr_u_bad
            local_sols_B[ap_i,4] = (local_sols_B[ap_i,2] - earnings)/(1+r)
            local_sols_B[ap_i,5] = 1
        else
            temp_vec = u_func.(local_sols_B[ap_i,2] .- a_grid_pos_ncr, σ) .+ V_hat_func.(1:a_size_pos_ncr, Ref(λ*V_good_p_pos_ncr+(1-λ)*V_bad_p_ncr), β, Ref(Px_i))
            if (ap_i - ncr_l_bad + 1) == findall(temp_vec .== maximum(temp_vec))[1]
                local_sols_B[ap_i,4] = (local_sols_B[ap_i,2] - earnings)/(1+r)
                local_sols_B[ap_i,5] = 1
            end
        end
    end

    # return global solutions
    return local_sols_B[local_sols_B[:,5] .== 1.0, 1:4]
end

function update_G_func!(x_i::Integer, V_good_repay::Array{Float64,2}, policy_a_good_repay::Array{Float64,2}, global_sols_G::Array{Float64,2}, a_grid::Array{Float64,1})
    # update the value and policy functions for the current type (x_i) of good credit history with full repayment (a_i)
    for a_i in 1:length(a_grid)
        # (1) value function
        V_G_func = LinearInterpolation(global_sols_G[:,4], global_sols_G[:,3], extrapolation_bc = Line())
        V_good_repay[a_i,x_i] = V_G_func(a_grid[a_i])

        # (2) policy function
        a_G_func = LinearInterpolation(global_sols_G[:,4], global_sols_G[:,1], extrapolation_bc = Line())
        policy_a_good_repay[a_i,x_i] = a_G_func(a_grid[a_i])
    end
end

function update_B_func!(x_i::Integer, V_bad::Array{Float64,2}, policy_a_bad::Array{Float64,2}, global_sols_B::Array{Float64,2}, a_grid_pos::Array{Float64,1})
    # update the value and policy functions for the current type (x_i) of bad credit history with savings (a_i)
    for a_i in 1:length(a_grid_pos)
        # (1) value function
        V_B_func = LinearInterpolation(global_sols_B[:,4], global_sols_B[:,3], extrapolation_bc = Line())
        V_bad[a_i,x_i] = V_B_func(a_grid_pos[a_i])

        # (2) policy function
        a_B_func = LinearInterpolation(global_sols_B[:,4], global_sols_B[:,1], extrapolation_bc = Line())
        policy_a_bad[a_i,x_i] = a_B_func(a_grid_pos[a_i])
    end
end

function households!(variables::mut_vars, parameters::NamedTuple; tol = 1E-8, iter_max = 100)
    # solve the household's maximization problem to obtain the converged value functions via the modified EGM by Fella (2014, JEDC), given price schedules

    # unpack parameters
    @unpack a_grid, a_grid_neg, a_grid_pos, ind_a_zero, x_grid, x_size, β, Px, λ, σ, ξ, r = parameters

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving household's maximization: ")

    # initialize the next-period value functions
    V_good_p = similar(variables.V_good)
    V_bad_p = similar(variables.V_bad)

    while crit > tol && iter < iter_max

        # copy the current value functions to the pre-specified containers
        copyto!(V_good_p, variables.V_good)
        copyto!(V_bad_p, variables.V_bad)

        # start looping over each household's type
        for x_i in 1:x_size

            # unpack or construct the individual states and variables
            p, t, e = x_grid[x_i,:]
            earnings = p*t - e
            q_i = variables.q[:,x_i]
            Px_i = Px[x_i,:]

            #--------------------------#
            # compute global solutions #
            #--------------------------#
            # (1) bad credit history

            # compute the non-concave region
            V_good_p_pos = V_good_p[ind_a_zero:end,:]
            ncr_l_bad, ncr_u_bad = ncr_func(λ*V_good_p_pos+(1-λ)*V_bad_p, a_grid_pos, β, Px_i)

            # compute global solutions
            global_sols_B = sols_B_func(ncr_l_bad, ncr_u_bad, V_good_p_pos, V_bad_p, a_grid_pos, β, Px_i, λ, σ, r, earnings)

            # (2) good credit history

            # compute the risky borrowing limit
            rbl, rbl_ind = rbl_func(q_i, a_grid_neg)

            # define the variables whose indices are above the risky borrowing limit
            a_grid_rbl = a_grid[rbl_ind:end]
            a_size_rbl = length(a_grid_rbl)
            V_good_p_rbl = V_good_p[rbl_ind:end,:]
            q_i_rbl = q_i[rbl_ind:end]

            # compute the non-concave region
            ncr_l_good, ncr_u_good = ncr_func(V_good_p_rbl, a_grid_rbl, β, Px_i)

            # compute global solutions
            global_sols_G = sols_G_func(ncr_l_good, ncr_u_good, V_good_p_rbl, a_grid_rbl, q_i_rbl, β, Px_i, σ, r, earnings)

            #-----------------------------------#
            # update value and policy functions #
            #-----------------------------------#
            # (1) bad credit history
            update_B_func!(x_i, variables.V_bad, variables.policy_a_bad, global_sols_B, a_grid_pos)

            # (2) good credit history with repayment
            update_G_func!(x_i, variables.V_good_repay, variables.policy_a_good_repay, global_sols_G, a_grid)

            # (3) good credit history with defaulting
            variables.V_good_default[:,x_i] .= u_func(p*t*(1-ξ), σ) + V_hat_func(1, V_bad_p, β, Px_i)

            # (4) good credit history
            variables.V_good[:,x_i] = max.(variables.V_good_repay[:,x_i], variables.V_good_default[:,x_i])
        end

        # check convergence
        crit = max(norm(variables.V_good-V_good_p, Inf), norm(variables.V_bad-V_bad_p, Inf))

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end

function banks!(variables::mut_vars, parameters::NamedTuple)
    # update the price schedule

    # unpack parameters
    @unpack ξ, r, a_grid, a_size_neg, Px, x_grid, x_size = parameters

    # update pricing function and default probability
    for x_i in 1:x_size
        for ap_i in 1:a_size_neg
            # compute the expected revenue and the associated bond price
            revenue_expect = 0
            for xp_i in 1:x_size
                pp, tp, ep = x_grid[xp_i,:]
                if variables.V_good_default[ap_i,xp_i] > variables.V_good_repay[ap_i,xp_i]
                    revenue_expect += Px[x_i,xp_i]*ξ*pp*tp
                else
                    revenue_expect += Px[x_i,xp_i]*(-a_grid[ap_i])
                end
            end
            q_update = revenue_expect / ( (1+r)*(-a_grid[ap_i]) )
            variables.q[ap_i,x_i] = q_update < (1/(1+r)) ? q_update : 1/(1+r)
        end
    end
end
