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

function rbl_func(x_i::Integer, q::Array{Float64,2}, a_grid_neg::Array{Float64,1})
    # compute the risky borrowing limit for curren type (x_i)
    a_size_neg = length(a_grid_neg)
    qa_func = LinearInterpolation(a_grid_neg, q[1:a_size_neg,x_i].*a_grid_neg)
    results = optimize(qa_func, a_grid_neg[1], 0)
    rbl = results.minimizer
    rbl_ind = minimum(findall(a_grid_neg .> rbl))
    return rbl, rbl_ind
end

function V_hat_func(ap_i::Integer, x_i::Integer, V::Array{Float64,2}, β::Real, Px::Array{Float64,2})
    # compute the discounted expected value function for asset holding in the next period (ap_i) and current type (x_i)
    return β*sum(Px[x_i,:].*V[ap_i,:])
end

function dV_hat_func(ap_i::Integer, x_i::Integer, V::Array{Float64,2}, a_grid::Array{Float64,1}, β::Real, Px::Array{Float64,2})
    # compute first-order derivative of the discounted expected value function for asset holding in the next period (ap_i) and current type (x_i) wrt the first argument (ap_i) by forward finite difference
    if ap_i < size(V,1)
        return (V_hat_func(ap_i+1,x_i,V,β,Px) - V_hat_func(ap_i,x_i,V,β,Px)) / (a_grid[ap_i+1] - a_grid[ap_i])
    else
        return (V_hat_func(ap_i,x_i,V,β,Px) - V_hat_func(ap_i-1,x_i,V,β,Px)) / (a_grid[ap_i] - a_grid[ap_i-1])
    end
end

function ncr_func(x_i::Integer, V::Array{Float64,2}, a_grid::Array{Float64,1}, β::Real, Px::Array{Float64,2})
    # compute the non-concave region for type (x_i)
    a_size = length(a_grid)
    ncr_l, ncr_u = 1, a_size

    # (1) find the lower bound
    dV_hat_vec = dV_hat_func.(1:a_size, x_i, V, a_grid, β, Px)
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

function CoH_G_func(ap_i::Integer, x_i::Integer, V_good::Array{Float64,2}, a_grid::Array{Float64,1}, q::Array{Float64,2}, β::Real, Px::Array{Float64,2})
    # compute the cash on hands for the case of asset holding in the next period (ap_i), current type (x_i), and good credit history with repayment. Note that cash on hands for the case of good credit history with defaulting is trivially determined so it can be ignored
    return inv_du_func(dV_hat_func(ap_i, x_i, V_good, a_grid, β, Px)/q[ap_i,x_i]) + q[ap_i,x_i]*a_grid[ap_i]
end

function CoH_B_func(ap_i::Integer, x_i::Integer, V_good::Array{Float64,2}, V_bad::Array{Float64,2}, a_grid_pos::Array{Float64,1}, q::Array{Float64,2}, β::Real, Px::Array{Float64,2}, λ::Real)
    # compute the cash on hands for asset holding in the next period (ap_i), current type (x_i), and bad credit hostory. Note that ap must be positive (saving only)
    return inv_du_func(λ*dV_hat_func(ap_i, x_i, V_good, a_grid_pos, β, Px)+(1-λ)*dV_hat_dunc(ap_i, x_i, V_bad, a_grid_pos, β, Px)) + a_grid_pos[ap_i]
end

function local_sols_G_func(x_i::Integer, a_grid_rbl::Array{Float64,1}, EV_good_rbl::Array{Float64,2}, q_rbl::Array{Float64,2})
    # create a matrix of local solutions (bounded below by the risky borrowing limit), including (1) next-period asset holdings, (2) cash on hands, (3) associated value functions, and (4) identifier of a global solution
    a_size_rbl = length(a_grid_rbl)
    local_sols_G = zeros(a_size_rbl,4)
    local_sols_G[:,1] = a_grid_rbl
    local_sols_G[:,2] = CoH_G_func.(1:a_size_rbl, x_i, EV_good_rbl, a_grid_rbl, q_rbl, β, Px)
    local_sols_G[:,3] = u.(local_sols_G[:,2] .- q_rbl[:,x_i].*local_sols_G[:,1]) .+ V_hat_func.(1:a_size_rbl, x_i, EV_good_rbl, β, Px)
    return local_sols_G
end

function global_sols_G_func(x_i::Integer, V::Array{Float64,2}, q::Array{Float64,2}, parameters::NamedTuple)
    # identify global solutions
    for ap_i in 1:a_size_eff
        if (ap_i+rbl_ind-1)<ncr_l || (ap_i+rbl_ind-1)>ncr_u
            local_sols[ap_i,4] = 1
        else
            u.(local_sols_G[ap_i,2] .- q_rbl[:,x_i].*a_grid[ncr_l:bcr_u]) .+ V_hat_func.(ncr_l:bcr_u, x_i, EV_good_rbl, β, Px)
        end
    end
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

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf

    # initialize the next-period value functions
    EV_good = similar(variables.V_good)
    EV_bad = similar(variables.V_bad)

    while crit > tol && iter < iter_max

        # copy the current value functions to the pre-specified containers
        copyto!(EV_good, variables.V_good)
        copyto!(EV_bad, variables.V_bad)

        # start looping over each household's type
        for x_i in 1:x_size

            # unpack the individual states and variables
            p, t, e = x_grid[x_i,:]

            #---------------------#
            # good credit history #
            #---------------------#
            # (1) repayment

            # compute the risky borrowing limit
            rbl, rbl_ind = rbl_func(x_i, variables.q, a_grid_neg)

            # compute the non-concave region
            ncr_l, ncr_u = ncr_func(x_i, EV_good, a_grid, β, Px)

            # check if the risky borrowing limit is lower than the lower bound of the non-concave region
            if rbl_ind > ncr_l
                println("WARNING: risky borrowing limit is greater than the lower bound of non-concave region")
                break
            end

            # define variables on the grids above the risky borrowing limit
            a_grid_rbl = a_grid[rbl_ind:end]
            EV_good_rbl = EV_good[rbl_ind:end,x_i]
            q_rbl = variables.q[rbl_ind:end,x_i]

            # compute local solutions
            local_sols_G = local_sols_G_func(x_i, a_grid_rbl, EV_good_rbl, q_rbl)

            # identify glocal solutions
            global_sols_G =



            # (2) defaulting


            #--------------------#
            # bad credit history #
            #--------------------#

    end
end
