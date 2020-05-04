function u_func(c::Real, σ::Real)
    # compute utility
    if c > 0
        if σ == 1
            return log(c)
        else
            return 1 / ((1-σ)*c^(σ-1))
        end
    else
        println("WARNING: non-positive consumption")
        return -Inf
    end
end

function du_func(c::Real, σ::Real)
    # compute marginal utility
    if c > 0
        return c^(-σ)
    else
        println("non-positive consumption!")
    end
end

function inv_du_func(x::Real, σ::Real)
    # compute marginal utility
    return x^(-1/σ)
end

function rbl_func(q_i::Array{Float64,1}, a_grid_neg::Array{Float64,1})
    # compute the risky borrowing limit for curren type (x_i)
    a_size_neg = length(a_grid_neg)
    qa_func = LinearInterpolation(a_grid_neg, q_i[1:a_size_neg] .* a_grid_neg)
    results = optimize(qa_func, a_grid_neg[1], 0)
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
        return (V_hat_func(ap_i,V,β,Px_i) - V_hat_func(ap_i-1,V,β,Px_i)) / (a_grid[ap_i] - a_grid[ap_i-1])
    end
end

function ncr_func(V_p::Array{Float64,2}, a_grid::Array{Float64,1}, β::Real, Px_i::Array{Float64,1})
    # compute the non-concave region for type (x_i)
    a_size = length(a_grid)
    ncr_l, ncr_u = 1, a_size

    # (1) find the lower bound
    dV_hat_vec = dV_hat_func.(1:a_size, V_p, a_grid, β, Px_i)
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

function CoH_G_func(ap_i::Integer, V_good_p::Array{Float64,2}, a_grid::Array{Float64,1}, q_i::Array{Float64,1}, β::Real, Px_i::Array{Float64,1})
    # compute the cash on hands for the case of asset holding in the next period (ap_i), current type (x_i), and good credit history with repayment. Note that cash on hands for the case of good credit history with defaulting is trivially determined so it can be ignored
    return inv_du_func(dV_hat_func(ap_i, V_good_p, a_grid, β, Px_i)/q_i[ap_i]) + q_i[ap_i]*a_grid[ap_i]
end

function CoH_B_func(ap_i::Integer, V_good_p_pos::Array{Float64,2}, V_bad_p::Array{Float64,2}, a_grid_pos::Array{Float64,1}, β::Real, Px_i::Array{Float64,1}, λ::Real)
    # compute the cash on hands for asset holding in the next period (ap_i), current type (x_i), and bad credit hostory. Note that ap must be positive (saving only)
    return inv_du_func(dV_hat_func(ap_i, λ*V_good_p_pos+(1-λ)*V_bad_p, a_grid_pos, β, Px_i)) + a_grid_pos[ap_i]
end

function sols_G_func(rbl_ind::Integer, ncr_l_good::Integer, ncr_u_good::Integer, V_good_p::Array{Float64,2}, a_grid::Array{Float64,1}, q_i::Array{Float64,1}, β::Real, Px_i::Array{Float64,1}, r::Real, earnings::Real)
    # (1) create a matrix of local solutions for good credit history with repayment (bounded below by the risky borrowing limit), including (a) next-period asset holdings, (b) cash on hands, (c) associated value functions, (d) associated current asset holdings, and (e) identifier of a global solution

    # define the variables whose indices are above the risky borrowing limit
    a_grid_rbl = a_grid[rbl_ind:end]
    a_size_rbl = length(a_grid_rbl)
    V_good_p_rbl = V_good_p[rbl_ind:end,:]
    q_i_rbl = q_i[rbl_ind:end]

    # construct the matrix storing all possible local solutions
    local_sols_G = zeros(a_size_rbl,5)
    local_sols_G[:,1] = a_grid_rbl
    local_sols_G[:,2] = CoH_G_func.(1:a_size_rbl, V_good_p_rbl, a_grid_rbl, q_i_rbl, β, Px_i)
    local_sols_G[:,3] = u.(local_sols_G[:,2] .- q_i_rbl .* local_sols_G[:,1]) .+ V_hat_func.(1:a_size_rbl, V_good_p_rbl, β, Px_i)

    # (2) identify global solutions by introduing an additional discretized VFI maximization step for the points in the non-cave region

    # define the variables whose indices are within the non-concave region
    a_grid_ncr = a_grid[ncr_l_good:ncr_u_good]
    a_size_ncr = length(a_grid_ncr)
    V_good_p_ncr = V_good_p[ncr_l_good:ncr_u_good,:]
    q_i_ncr = q_i[ncr_l_good:ncr_u_good]

    # construct the matrix storing global solutions
    for ap_i in 1:a_size_rbl
        if (ap_i+rbl_ind-1)<ncr_l_good || (ap_i+rbl_ind-1)>ncr_u_good
            if local_sols_G[ap_i,2] - earnings >= 0
                local_sols_G[ap_i,4] = (local_sols_G[ap_i,2] - earnings)/(1+r)
                local_sols_G[ap_i,5] = 1
            else
                local_sols_G[ap_i,4] = local_sols_G[ap_i,2] - earnings
                local_sols_G[ap_i,5] = 1
            end
        else
            temp_vec = u.(local_sols_G[ap_i,2] .- q_i_ncr .* a_grid_ncr) .+ V_hat_func.(1:a_size_ncr, V_good_p_ncr, β, Px_i)
            if ap_i == findall(temp_vec .== maximum(temp_vec))[1]
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

function sols_B_func(ncr_l_bad::Integer, ncr_u_bad::Integer, V_good_p_pos::Array{Float64,2}, V_bad_p::Array{Float64,2}, a_grid_pos::Array{Float64,1}, β::Real, Px_i::Array{Float64,1}, λ::Real, r::Real, earnings::Real)
    # (1) create a matrix of local solutions for bad credit history, including (a) next-period asset holdings, (b) cash on hands, (c) associated value functions, (d) associated current asset holdings, and (e) identifier of a global solution

    # construct the matrix storing all possible local solutions
    a_size_pos = length(a_grid_pos)
    local_sols_B = zeros(a_size_pos,5)
    local_sols_B[:,1] = a_grid_pos
    local_sols_B[:,2] = CoH_B_func.(1:a_size_pos, V_good_p_pos, V_bad_p, a_grid_pos, β, Px_i, λ)
    local_sols_B[:,3] = u.(local_sols_B[:,2] .- local_sols_B[:,1]) .+ V_hat_func.(1:a_size_pos, λ*V_good_p_pos+(1-λ)*V_bad_p, β, Px_i)

    # (2) identify global solutions by introduing an additional discretized VFI maximization step for the points in the non-cave region

    # define the variables whose indices are within the non-concave region
    a_grid_pos_ncr = a_grid_pos[ncr_l_bad:ncr_u_bad]
    a_size_pos_ncr = length(a_grid_pos_ncr)
    V_good_p_pos_ncr = V_good_p_pos[ncr_l_bad:ncr_u_bad,:]
    V_bad_p_ncr = V_bad_p[ncr_l_bad:ncr_u_bad,:]

    # construct the matrix storing global solutions
    for ap_i in 1:length(a_grid_pos)
        if ap_i<ncr_l_bad || ap_i>ncr_u_bad
            local_sols_B[ap_i,4] = (local_sols_B[ap_i,2] - earnings)/(1+r)
            local_sols_B[ap_i,5] = 1
        else
            temp_vec = u.(local_sols_B[ap_i,2] .- a_grid_pos_ncr) .+ V_hat_func.(1:a_size_pos_ncr, λ*V_good_p_pos_ncr+(1-λ)*V_bad_p_ncr, β, Px_i)
            if ap_i == findall(temp_vec .== maximum(temp_vec))[1]
                local_sols_B[ap_i,4] = (local_sols_B[ap_i,2] - earnings)/(1+r)
                local_sols_B[ap_i,5] = 1
            end
        end
    end

    # return global solutions
    return local_sols_B[local_sols_B[:,5] .== 1.0, 1:4]
end

function update_G_func!(variables.V_good_repay::Array{Float64,2}, variables.policy_good_repay::Array{Float64,2}, global_sols_G::Array{Float64,2}, a_i::Integer, a_grid::Array{Float64,1})
    # update the value and policy functions for the current type (x_i) of good credit history with full repayment (a_i)

    # (1) value function
    V_G_func = LinearInterpolation(global_sols_G[:,4], global_sols_G[:,3], extrapolation_bc = Line())
    variables.V_good_repay[a_i,x_i] = V_G_func(a_grid[a_i])

    # (2) policy function
    a_G_func = LinearInterpolation(global_sols_G[:,4], global_sols_G[:,1], extrapolation_bc = Line())
    variables.policy_good_repay[a_i,x_i] = a_G_func(a_grid[a_i])
end

function update_B_func(variables.V_bad::Array{Float64,2}, variables.policy_bad::Array{Float64,2}, global_sols_B::Array{Float64,2}, a_i::Integer, a_grid_pos::Array{Float64,1})
    # update the value and policy functions for the current type (x_i) of bad credit history with savings (a_i)

    # (1) value function
    V_B_func = LinearInterpolation(global_sols_B[:,4], global_sols_B[:,3], extrapolation_bc = Line())
    variables.V_bad[a_i,x_i] = V_G_func(a_grid_pos[a_i])

    # (2) policy function
    a_B_func = LinearInterpolation(global_sols_B[:,4], global_sols_B[:,1], extrapolation_bc = Line())
    variables.policy_bad[a_i,x_i] = a_G_func(a_grid_pos[a_i])
end

function households(variables::mut_vars, parameters::NamedTuple; tol = 1E-8, iter_max = 100)
    # solve the household's maximization problem to obtain the converged value functions via the modified EGM by Fella (2014, JEDC), given price schedules

    # unpack parameters
    @unpack = parameters
    a_grid
    a_grid_neg
    a_size_neg
    β
    Px
    ξ
    r

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf

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
            global_sols_B = sols_B_func(ncr_l_bad:, ncr_u_bad, V_good_p_pos, V_bad_p, a_grid_pos, β, Px_i, λ, r, earnings)

            # (2) good credit history

            # compute the risky borrowing limit
            rbl, rbl_ind = rbl_func(q_i, a_grid_neg)

            # compute the non-concave region
            ncr_l_good, ncr_u_good = ncr_func(V_good_p, a_grid, β, Px_i)

            # check if the risky borrowing limit is lower than the lower bound of the non-concave region
            if rbl_ind > ncr_l
                println("WARNING: risky borrowing limit is greater than the lower bound of non-concave region")
                break
            end

            # compute global solutions
            global_sols_G = sols_G_func(rbl_ind, ncr_l_good, ncr_u_good, V_good_p, a_grid, q_i, β, Px_i, r, earnings)

            #-----------------------------------#
            # update value and policy functions #
            #-----------------------------------#
            # (1) bad credit history
            variables.V_bad[:,x_i] =

            # (2) good credit history with repayment
            variables.V_good_repay[:,x_i] =
            variables.policy_a_good_default

            # (3) good credit history with defaulting
            variables.V_good_default[:,x_i] .= u(p*t*(1-ξ)) + V_hat_func(1, V_bad_p, β, Px_i)
        end

        crit = norm(NR_variables.μ-Eμ, Inf)
        ProgressMeter.update!(prog_μ, crit_μ)
    end
end
