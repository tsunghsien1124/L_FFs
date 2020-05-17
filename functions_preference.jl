function para(; λ::Real = 0.10,         # history rased probability
                β::Real = 0.96,         # discount factor
                ξ::Real = 0.25,         # garnishment rate
                σ::Real = 2,            # CRRA coefficient
                r_f::Real = 0.03,       # risk-free rate
                ρ_p::Real = 0.95,       # AR(1) of persistent shock
                σ_p::Real = 0.05,       # s.d. of persistent shock
                p_size::Integer = 7,    # no. of persistent shock
                ν_size::Integer = 2,    # no. of preference shock
                a_min::Real = -1,       # min of asset holding
                a_max::Real = 10,       # max of asset holding
                a_size::Integer = 100)  # number of the grid asset holding
    #------------------------------------------------------#
    # contruct an immutable object containg all paramters. #
    #------------------------------------------------------#

    # persistent shock
    Mp = tauchen(p_size, ρ_p, σ_p)
    Pp = Mp.p
    p_grid = collect(Mp.state_values) .+ 1.0

    # preference schock
    ν_grid = [0.7, 1]
    Pν = repeat([0.10 0.90], ν_size, 1)

    # idiosyncratic transition matrix
    Px = kron(Pν, Pp)
    x_grid = gridmake(p_grid, ν_grid)
    x_ind = gridmake(1:p_size, 1:ν_size)
    x_size = p_size*ν_size

    # asset holding grid
    a_mode = 0
    if a_mode == 0
        a_size_neg = convert(Int, a_size/2)
        a_grid_neg = collect(range(a_min, 0, length = a_size_neg))
        a_size_pos = a_size - a_size_neg
        a_grid_pos = collect(range(0, a_max, length = a_size_pos))
        a_grid = cat(a_grid_neg, a_grid_pos[2:end], dims = 1)
        a_size = length(a_grid)
        ind_a_zero = findall(a_grid .== 0)[1]
    else
        a_degree = 1
        a_grid = ((range(0, stop = a_size-1, length = a_size)/(a_size-1)).^a_degree)*(a_max-a_min) .+ a_min
        ind_a_zero = findall(a_grid .>= 0)[1]
        a_grid[ind_a_zero] = 0
        a_size_neg = ind_a_zero
        a_size_pos = a_size - a_size_neg + 1
        a_grid_neg = a_grid[1:ind_a_zero]
        a_grid_pos = a_grid[ind_a_zero:end]
    end

    # return values
    return (λ = λ, β = β, ξ = ξ, σ = σ, r_f = r_f, a_grid = a_grid, ind_a_zero = ind_a_zero, a_size = a_size, a_size_pos = a_size_pos, a_size_neg = a_size_neg, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos, Pp = Pp, p_grid = p_grid, p_size = p_size, Pν = Pν, ν_grid = ν_grid, ν_size = ν_size, Px = Px, x_grid = x_grid, x_size = x_size, x_ind = x_ind)
end

mutable struct mut_vars
    q::Array{Float64,3}
    V_good::Array{Float64,3}
    V_bad::Array{Float64,2}
    policy_a_good::Array{Float64,3}
    policy_a_bad::Array{Float64,2}
    global_sols_good::Array{Float64,3}
    global_sols_bad::Array{Float64,3}
end

function vars(parameters::NamedTuple)
    #-------------------------------------------------------------#
    # construct a mutable object containing endogenous variables. #
    #-------------------------------------------------------------#

    # unpack parameters
    @unpack β, σ, r_f, a_grid, a_grid_neg, a_grid_pos, a_size, a_size_neg, a_size_pos, ind_a_zero, x_grid, x_size, p_grid = parameters

    # define pricing related variables
    # (1: price, 2: price derivative, 3: size, 4: size derivative)
    q = zeros(a_size, x_size, 4)
    q_mode = 0
    if q_mode == 0
        q[:,:,1] .= 1.0
        q[1:ind_a_zero,:,1] .= 1/(1+r_f)
    else
        q[:,:,1] .= 1.0
        for x_ind in 1:x_size
            p_i, ν_i = x_grid[x_ind,:]
            for a_ind in 1:ind_a_zero
                q[a_ind,x_ind,1] = (p_i/p_grid[end]) * ((a_grid[1]-a_grid[a_ind])/a_grid[1]) / (1 + r_f)
            end
        end
    end
    q[:,:,2] = derivative_func(a_grid, q[:,:,1])
    q[:,:,3] = q[:,:,1] .* repeat(a_grid, 1, x_size)
    q[:,:,4] = derivative_func(a_grid, q[:,:,3])

    # define value functions (1: good, 2: good and repay, 3: good but default)
    V_good = zeros(a_size, x_size, 3)
    for x_ind in 1:x_size
        p, ν = x_grid[x_ind,:]
        for a_ind in 1:a_size
            a = a_grid[a_ind]
            V_good[a_ind,x_ind,1] = a < 0 ? u_func(p+(r_f/(1+r_f))*a, σ) / (1-ν*β) : u_func(p+r_f*a, σ) / (1-ν*β)
        end
    end
    V_bad = V_good[ind_a_zero:end,:,1]

    # define policy functions
    policy_a_good = zeros(a_size, x_size, 3)
    policy_a_bad = zeros(a_size_pos, x_size)

    # define local solutions
    global_sols_good = zeros(a_size_pos, 7, x_size)
    global_sols_bad = zeros(a_size_pos, 7, x_size)

    # return outputs
    variables = mut_vars(q, V_good, V_bad, policy_a_good, policy_a_bad, global_sols_good, global_sols_bad)
    return variables
end

function u_func(c::Real, σ::Real; method::Integer = 0)
    #---------------------------------------------------------------------#
    # compute utility, marginal utility, and inverse of marginal utility. #
    #---------------------------------------------------------------------#

    # compute utility (method = 0)
    if method == 0
        if c > 0
            return σ == 1 ? log(c) : 1 / ((1-σ)*c^(σ-1))
        else
            # println("WARNING: utility of non-positive consumption!")
            return -Inf
        end
    # marginal utility (method = 1)
    elseif method == 1
        return c > 0 ? 1/(c^σ) : println("WARNING: marginal utility of non-positive consumption!")
    # inverse of marginal utility (method = 2)
    else
        return c > 0 ? 1/(c^(1/σ)) : println("WARNING: inverse of non-positive marginal utility!")
    end
end

function V_hat_func(β::Real, Px_i::Array{Float64,1}, V_p::Array{Float64,2})
    #-------------------------------------------------#
    # compute the discounted expected value function. #
    #-------------------------------------------------#
    return β*V_p*Px_i
end

function derivative_func(x::Array{Float64,1}, y; method::Integer = 0)
    #------------------------------------------------#
    # compute (approximated) first-order derivative. #
    #------------------------------------------------#
    x_size = length(x)
    if isa(y, Array{Float64,1})
        y_size_r, y_size_c = length(y), 1
        dy = zeros(x_size)
    else
        y_size_r, y_size_c = size(y)
        dy = zeros(x_size, y_size_c)
    end
    if x_size == y_size_r
        # forward finite difference (method = 0)
        if method == 0
            for y_ind_c in 1: y_size_c, x_ind in 1:x_size
                dy[x_ind,y_ind_c] = x_ind < x_size ? (y[x_ind+1,y_ind_c] - y[x_ind,y_ind_c]) / (x[x_ind+1] - x[x_ind]) : dy[x_ind-1,y_ind_c]
            end
            return dy
        # backward finite difference (method = 1)
        else
            for y_ind_c in 1: y_size_c, x_ind in x_size:-1:1
                dy[x_ind,y_ind_c] = x_ind > 1 ? (y[x_ind,y_ind_c] - y[x_ind-1,y_ind_c]) / (x[x_ind] - x[x_ind-1]) : dy[x_ind+1,y_ind_c]
            end
            return dy
        end
    else
        println("WARNING: size mismatch! x_size = $x_size but y_size_r = $y_size_r")
        return nothing
    end
end

function rbl_func(V_p::Array{Float64,2}, q_i::Array{Float64,2}, a_grid::Array{Float64,1})
    #--------------------------------#
    # compute risky borrowing limit. #
    #--------------------------------#
    Dqa_check = Inf
    Dqa_iter = length(a_grid)
    while Dqa_check > 0
        Dqa_check = q_i[Dqa_iter, 4]
        Dqa_check = q_i[Dqa_iter, 4] <= 0 ? break : Dqa_iter -= 1
    end
    rbl_ind = Dqa_iter + 1
    rbl = a_grid[rbl_ind]
    V_p_rbl = V_p[rbl_ind:end,:]
    q_i_rbl = q_i[rbl_ind:end,:]
    a_grid_rbl = a_grid[rbl_ind:end]
    a_size_rbl = length(a_grid_rbl)
    return rbl_ind, rbl, V_p_rbl, q_i_rbl, a_grid_rbl, a_size_rbl
end

function ncr_func(DV_hat::Array{Float64,1})
    #---------------------------------#
    # compute the non-concave region. #
    #---------------------------------#
    a_size = length(DV_hat)
    ncr_l, ncr_u = 1, a_size
    # (1) find the lower bound
    V_max, i_max = DV_hat[1], 1
    while i_max < a_size
        if V_max > maximum(DV_hat[(i_max+1):end])
            i_max += 1
            V_max = DV_hat[i_max]
        else
            ncr_l = i_max
            break
        end
    end
    # (2) find the upper bound
    V_min, i_min = DV_hat[end], a_size
    while i_min > 1
        if V_min < minimum(DV_hat[1:(i_min-1)])
            i_min -= 1
            V_min = DV_hat[i_min]
        else
            ncr_u = i_min
            break
        end
    end
    return ncr_l, ncr_u
end

function CoH_func(V_hat::Array{Float64,1}, q_i::Array{Float64,2}, a_grid::Array{Float64,1}, σ::Real; check_zero::Integer = 0, check_mode::Integer = 0)
    #----------------------------#
    # compute the cash on hands. #
    #----------------------------#
    V_size, q_size_r, a_size = length(V_hat), size(q_i,1), length(a_grid)
    allsize = [V_size, q_size_r, a_size]
    allsame(x) = all(y -> y==x[1], x)
    if allsame(allsize)
        DV_hat = derivative_func(a_grid, V_hat)
        CoH = u_func.(DV_hat ./ q_i[:,4], σ; method = 2) .+ q_i[:,3]
        if check_zero == 1
            ind_a_zero = findall(a_grid .== 0)[1]
            if check_mode == 0
                CoH[ind_a_zero] = u_func(DV_hat[ind_a_zero], σ; method = 2)
            else
                obj_zero(x) = u_func(x, σ) + V_hat[ind_a_zero] - u_func(x - q_i[ind_a_zero+1,3], σ) - V_hat[ind_a_zero+1]
                CoH[ind_a_zero] = find_zero(obj_zero, CoH[ind_a_zero+1])
            end
        end
        return CoH
    else
        println("WARNING: size mismatch! V_size = $V_size, q_size_r = $q_size_r, and a_size = $a_size.")
        return nothing
    end
end

function sols_func(β::Real, Px_i::Array{Float64,1}, V_p::Array{Float64,2}, a_grid::Array{Float64,1}, q_i::Array{Float64,2}, σ::Real, r_f::Real, earnings::Real; check_rbl::Integer = 0)
    # (1) return global solutions for non-negative asset holdings.
    # contrcut variables above zero asset holding
    a_size = length(a_grid)
    ind_a_zero = findall(a_grid .== 0)[1]
    a_grid_pos = a_grid[ind_a_zero:end]
    a_size_pos = length(a_grid_pos)
    V_p_pos = V_p[ind_a_zero:end,:]
    q_i_pos = q_i[ind_a_zero:end,:]
    V_hat_pos = V_hat_func(β, Px_i, V_p_pos)

    # construct the matrix containg all possible "positive" local solutions
    local_sols = zeros(a_size_pos, 8)
    local_sols[:,1] = a_grid_pos
    local_sols[:,2] = q_i_pos[:,1]
    local_sols[:,3] = q_i_pos[:,3]
    local_sols[:,4] = V_hat_pos
    local_sols[:,5] = CoH_func(V_hat_pos, q_i_pos, a_grid_pos, σ; check_zero = check_rbl)
    local_sols[:,6] = u_func.(local_sols[:,5] .- local_sols[:,3], σ) .+ local_sols[:,4]

    # compute the non-concave region
    ncr_l, ncr_u = ncr_func(derivative_func(a_grid_pos, V_hat_pos))

    # define the variables whose indices are within the non-concave region
    a_grid_pos_ncr = a_grid_pos[ncr_l:ncr_u]
    q_i_pos_ncr = q_i_pos[ncr_l:ncr_u,:]
    V_hat_pos_ncr = V_hat_pos[ncr_l:ncr_u]

    # mark global pairs and compute associated current asset position
    for ap_ind in 1:a_size_pos
        a_temp = local_sols[ap_ind,5] - earnings
        if ap_ind < ncr_l || ap_ind > ncr_u
            local_sols[ap_ind,7] = a_temp / (1+r_f*(a_temp>=0))
            local_sols[ap_ind,8] = 1
        else
            V_temp = u_func.(local_sols[ap_ind,5] .- q_i_pos_ncr[:,3], σ) .+ V_hat_pos_ncr
            ind_max_V_temp = findall(V_temp .== maximum(V_temp))[1]
            if (ap_ind - ncr_l + 1) == ind_max_V_temp
                local_sols[ap_ind,7] = a_temp / (1+r_f*(a_temp>=0))
                local_sols[ap_ind,8] = 1
            end
        end
    end

    # export "positive" global solutions
    global_sols = local_sols[local_sols[:,8] .== 1.0, 1:7]

    # if holding zero asset holding is NOT included, make it!
    if global_sols[1,1] != a_grid_pos[1]
        # locate the first point in the global solutions
        ind_a_one = findall(a_grid_pos .== global_sols[1,1])[1]
        # define the objective function as in equation (21) in Fella's paper
        obj_C_0(x) = u_func(x-q_i_pos[1,3], σ) + V_hat_pos[1] - u_func(x-q_i_pos[ind_a_one,3], σ) - V_hat_pos[ind_a_one]
        C_0 = find_zero(obj_C_0, (q_i_pos[1,3], global_sols[1,5]))
        a_0 = (C_0-earnings) / (1+r_f*(C_0-earnings>0))
        V_0 = u_func(C1-q_i_pos[1,3], σ) + V_hat_pos[1]
        # expand the original global pairs
        global_sols = cat([a_grid_pos[1] q_i_pos[1,1] q_i_pos[1,3] V_hat_pos[1] C_0 V_0 a_0], global_sols, dims = 1)
    end

    # (2) update value and policy functions.
    # construct containers
    V = zeros(a_size)
    policy_a = zeros(a_size)

    if check_rbl == 0
        # define interpolated functions
        V_func = LinearInterpolation(global_sols[:,7], global_sols[:,6], extrapolation_bc = Line())
        a_func = LinearInterpolation(global_sols[:,7], global_sols[:,1], extrapolation_bc = Line())

        # extrapolate value and policy functions
        for a_ind in 1:a_size
            if a_grid[a_ind] >= global_sols[1,7]
                V[a_ind] = V_func(a_grid[a_ind])
                policy_a[a_ind] = a_func(a_grid[a_ind])
            else
                # compute the value as in equation (20) in Fella's paper
                V[a_ind] = u_func(earnings + (1+r_f*(a_grid[a_ind]>=0))*a_grid[a_ind] - global_sols[1,3], σ) + global_sols[1,4]
                policy_a[a_ind] = global_sols[1,1]
            end
        end
    else
        # compute the risky borrowing limit and define associated variables
        rbl_ind, rbl, V_p_rbl, q_i_rbl, a_grid_rbl, a_size_rbl = rbl_func(V_p, q_i, a_grid)
        ind_a_zero_rbl = findall(a_grid_rbl .== 0)[1]
        V_hat_rbl = V_hat_func(β, Px_i, V_p_rbl)

        # define interpolated functions above a' = 0
        V_func = LinearInterpolation(global_sols[:,7], global_sols[:,6], extrapolation_bc = Line())
        a_func = LinearInterpolation(global_sols[:,7], global_sols[:,1], extrapolation_bc = Line())

        # extrapolate value and policy functions
        for a_ind in 1:a_size
            if a_grid[a_ind] >= global_sols[1,7]
                V[a_ind] = V_func(a_grid[a_ind])
                policy_a[a_ind] = a_func(a_grid[a_ind])
            else
                # compute the value as in equation (20) in Jang and Lee's paper
                V_temp = u_func.(earnings .+ (1+r_f*(a_grid[a_ind]>=0))*a_grid[a_ind] .- q_i_rbl[1:ind_a_zero_rbl,3], σ) .+ V_hat_rbl[1:ind_a_zero_rbl]
                ind_a_temp = findall(V_temp .== maximum(V_temp))[1]
                V[a_ind] = V_temp[ind_a_temp]
                policy_a[a_ind] = a_grid_rbl[ind_a_temp]
            end
        end
    end

    # return results
    return global_sols, V, policy_a
end

function price!(variables::mut_vars, parameters::NamedTuple)
    #-------------------------------------------------------#
    # update the price schedule and associated derivatives. #
    #-------------------------------------------------------#
    @unpack ξ, r_f, a_grid, a_grid_neg, a_size_neg, ind_a_zero, Px, x_grid, x_size = parameters
    α = 0.15    # parameter controling update speed
    for x_ind in 1:x_size
        for ap_ind in 1:a_size_neg
            revenue = 0
            if ap_ind != ind_a_zero
                for xp_ind in 1:x_size
                    pp_i, νp_i = x_grid[xp_ind,:]
                    earnings = pp_i
                    if variables.V_good[ap_ind,xp_ind,3] >= variables.V_good[ap_ind,xp_ind,2]
                        revenue += Px[x_ind,xp_ind]*ξ*earnings
                    else
                        revenue += Px[x_ind,xp_ind]*(-a_grid_neg[ap_ind])
                    end
                end
                q_update = α*(revenue / ((1+r_f)*(-a_grid_neg[ap_ind]))) + (1-α)*variables.q[ap_ind,x_ind,1]
                variables.q[ap_ind,x_ind,1] = q_update < (1/(1+r_f)) ? q_update : 1/(1+r_f)
            else
                variables.q[ap_ind,x_ind,1] = 1/(1+r_f)
            end
        end
    end
    variables.q[:,:,2] = derivative_func(a_grid, variables.q[:,:,1])
    variables.q[:,:,3] = variables.q[:,:,1] .* repeat(a_grid, 1, x_size)
    variables.q[:,:,4] = derivative_func(a_grid, variables.q[:,:,3])
end

function solution!(variables::mut_vars, parameters::NamedTuple; tol = 1E-8, iter_max = 10000)
    # solve the household's maximization problem to obtain the converged value functions via the modified EGM by Fella (2014, JEDC), given price schedules

    # unpack parameters
    @unpack a_grid, a_size, a_grid_neg, a_grid_pos, ind_a_zero, x_grid, x_size, β, Px, λ, σ, ξ, r_f = parameters

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving household's maximization: ")

    # initialize the next-period value functions
    V_good_p = similar(variables.V_good[:,:,1])
    V_bad_p = similar(variables.V_bad)

    while crit > tol && iter < iter_max

        # println("iter = $iter")
        # copy the current value functions to the pre-specified containers
        copyto!(V_good_p, variables.V_good[:,:,1])
        copyto!(V_bad_p, variables.V_bad)

        # start looping over each household's type
        for x_ind in 1:x_size

            # abstract necessary variables
            Px_i = Px[x_ind,:]
            p_i, ν_i = x_grid[x_ind,:]
            q_i = variables.q[:,x_ind,:]

            # define two handy variables
            earnings = p_i
            β_adj = ν_i*β

            #-------------------------------------------------------------#
            # compute global solutions, update value and policy functions #
            #-------------------------------------------------------------#
            # (1) bad credit history
            # println("x_ind = $x_ind, bad")
            V_good_p_pos = V_good_p[ind_a_zero:end,:]
            q_i_pos = q_i[ind_a_zero:end,:]
            global_sols_bad, variables.V_bad[:,x_ind], variables.policy_a_bad[:,x_ind] = sols_func(β_adj, Px_i, λ*V_good_p_pos .+ (1-λ)*V_bad_p, a_grid_pos, q_i_pos, σ, r_f, earnings)

            # (2) good credit history with repayment
            # println("x_ind = $x_ind, good")
            global_sols_good, variables.V_good[:,x_ind,2], variables.policy_a_good[:,x_ind,2] = sols_func(β_adj, Px_i, V_good_p, a_grid, q_i, σ, r_f, earnings; check_rbl = 1)

            # (3) good credit history with defaulting
            V_hat_bad = V_hat_func(β_adj, Px_i, V_bad_p)
            variables.V_good[:,x_ind,3] .= u_func(earnings*(1-ξ), σ) + V_hat_bad[1]

            # (4) good credit history
            variables.V_good[:,x_ind,1] = max.(variables.V_good[:,x_ind,2], variables.V_good[:,x_ind,3])
            variables.policy_a_good[:,x_ind,1] = [variables.V_good[a_ind,x_ind,2] >= variables.V_good[a_ind,x_ind,3] ? variables.policy_a_good[a_ind,x_ind,2] : variables.policy_a_good[a_ind,x_ind,3] for a_ind in 1:a_size]
        end

        # update price, its derivative, and size of bond
        price!(variables, parameters)

        # check convergence
        crit = max(norm(variables.V_good[:,:,1]-V_good_p, Inf), norm(variables.V_bad-V_bad_p, Inf))

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end
