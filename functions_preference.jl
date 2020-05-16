function para(; λ::Real = 0.10,         # history rased probability
                β::Real = 0.96,         # discount factor
                ξ::Real = 0.25,         # garnishment rate
                σ::Real = 2,            # CRRA coefficient
                r_f::Real = 0.03,       # risk-free rate
                ρ_p::Real = 0.90,       # AR(1) of persistent shock
                σ_p::Real = 0.05,       # s.d. of persistent shock
                p_size::Integer = 15,    # no. of persistent shock
                ν_size::Integer = 2,    # no. of preference shock
                a_min::Real = -1,       # min of asset holding
                a_max::Real = 10,       # max of asset holding
                # a_scale::Integer = 10,  # scale of the grid asset holding
                a_size::Integer = 500,   # number of the grid asset holding
                a_degree::Integer = 1)  # degree of the grid asset holding

    # contruct an immutable object containg all paramters.

    # persistent shock
    Mp = tauchen(p_size, ρ_p, σ_p)
    Pp = Mp.p
    p_grid = collect(Mp.state_values) .+ 1.0

    # preference schock
    ν_grid = [0.8, 1]
    Pν = repeat([0.10 0.90], ν_size, 1)

    # idiosyncratic transition matrix conditional
    Px = kron(Pν, Pp)
    x_grid = gridmake(p_grid, ν_grid)
    x_ind = gridmake(1:p_size, 1:ν_size)
    x_size = p_size*ν_size

    # asset holding grid
    # a_size = convert(Int, (a_max-a_min)*a_scale + 1)
    # a_grid = ((range(0, stop = a_size-1, length = a_size)/(a_size-1)).^a_degree)*(a_max-a_min) .+ a_min
    a_size_neg = convert(Int, a_size/2)
    a_grid_neg = collect(range(a_min, 0, length = a_size_neg))
    a_grid_neg = a_grid_neg[1:(end-1)]
    a_size_neg = length(a_grid_neg)
    a_size_pos = convert(Int, a_size/2)
    a_grid_pos = collect(range(0, a_max, length = a_size_pos))
    a_grid = cat(a_grid_neg, a_grid_pos, dims = 1)
    a_size = length(a_grid)

    # find the index where a = 0
    ind_a_zero = findall(a_grid .>= 0)[1]
    a_grid[ind_a_zero] = 0

    # define the size of negative and positive asset
    a_size_neg = ind_a_zero - 1
    a_size_pos = a_size - a_size_neg

    # define the negative and positive asset holding grid
    a_grid_neg = a_grid[1:a_size_neg]
    a_grid_pos = a_grid[ind_a_zero:end]

    # return values
    return (λ = λ, β = β, ξ = ξ, σ = σ, r_f = r_f, a_grid = a_grid, ind_a_zero = ind_a_zero, a_size = a_size, a_size_pos = a_size_pos, a_size_neg = a_size_neg, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos, Pp = Pp, p_grid = p_grid, p_size = p_size, Pν = Pν, ν_grid = ν_grid, ν_size = ν_size, Px = Px, x_grid = x_grid, x_size = x_size, x_ind = x_ind)
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
    policy_matrix_a_good::SparseMatrixCSC{Float64,Int64}
    policy_matrix_a_good_default::SparseMatrixCSC{Float64,Int64}
    policy_matrix_a_good_repay::SparseMatrixCSC{Float64,Int64}
    local_sols_bad::Array{Float64,3}
    local_sols_good::Array{Float64,3}
    transition_matrix::SparseMatrixCSC{Float64,Int64}
    q::Array{Float64,2}
    μ::Array{Float64,3}
    QB::Real
    D::Real
end

function vars(parameters::NamedTuple)
    # construct a mutable object containing endogenous variables.

    # unpack parameters
    @unpack β, ξ, σ, r_f, a_grid, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos, ind_a_zero, x_grid, x_size, p_grid, p_size, ν_grid = parameters

    # define pricing function and default probability
    q_mode = 1
    if q_mode == 0
        q = ones(a_size, x_size)
        q[1:a_size_neg,:] .= 1/(1+r_f)
    else
        q = ones(a_size, x_size)
        for x_ind in 1:x_size
            p, ν = x_grid[x_ind,:]
            for a_ind in 1:a_size_neg
                q[a_ind,x_ind] = (p/p_grid[end]) * ((a_grid[1]-a_grid[a_ind])/a_grid[1]) / (1 + r_f)
            end
        end
    end

    # define value functions
    V_good = zeros(a_size, x_size)
    for x_ind in 1:x_size
        p, ν = x_grid[x_ind,:]
        for a_ind in 1:a_size
            a = a_grid[a_ind]
            V_good[a_ind,x_ind] = a < 0 ? u_func(p+(r_f/(1+r_f))*a, σ) / (1-ν*β) : u_func(p+r_f*a, σ) / (1-ν*β)
            # V_good[a_ind,x_ind] = u_func(p+r_f*a, σ) / (1-ν*β)
        end
    end
    V_good_repay = zeros(a_size, x_size)
    V_good_default = zeros(a_size, x_size)
    V_bad = V_good[ind_a_zero:end,:]

    # define policy functions
    policy_a_bad = zeros(a_size_pos, x_size)
    policy_a_good = zeros(a_size, x_size)
    policy_a_good_default = repeat([ind_a_zero], a_size, x_size)
    policy_a_good_repay = zeros(a_size, x_size)

    # define policy matrices
    policy_matrix_a_bad = spzeros(a_size, a_size*x_size)
    policy_matrix_a_good = spzeros(a_size, a_size*x_size)
    policy_matrix_a_good_default = spzeros(a_size, a_size*x_size)
    policy_matrix_a_good_repay = spzeros(a_size, a_size*x_size)

    # define local solutions
    local_sols_bad = zeros(a_size_pos, 7, x_size)
    local_sols_good = zeros(a_size, 7, x_size)

    # define the transition matrix for the cross-sectional distribution
    G_size = (a_size + a_size_pos) * x_size
    transition_matrix = spzeros(G_size, G_size)

    # initialize aggregate objects
    QB = 0.0
    D = 0.0

    # define stationary distribution
    μ = zeros(a_size, x_size, 2)
    μ[:,:,1] .= 1 / ( (a_size_pos+a_size+1) * x_size )
    μ[a_size_neg:end,:,2] .= 1 / ( (a_size_pos+a_size+1) * x_size )

    # return outputs
    variables = mut_vars(V_bad, V_good, V_good_default, V_good_repay, policy_a_bad, policy_a_good, policy_a_good_default, policy_a_good_repay, policy_matrix_a_bad, policy_matrix_a_good, policy_matrix_a_good_default, policy_matrix_a_good_repay, local_sols_bad, local_sols_good, transition_matrix, q, μ, QB, D)
    return variables
end

function u_func(c::Real, σ::Real; method::Integer = 0)
    # compute utility (method = 0), marginal utility (method = 1), inverse of marginal utility (method = 2).
    if method == 0
        if c > 0
            return σ == 1 ? log(c) : 1 / ((1-σ)*c^(σ-1))
        else
            return -Inf # println("WARNING: non-positive consumption")
        end
    elseif method == 1
        return c > 0 ? 1/(c^σ) : println("WARNING: non-positive consumption!")
    else
        return 1/(c^(1/σ))
        # return du > 0 ? 1/(du^(1/σ)) : println("WARNING: non-positive derivative of value function!")
    end
end

function V_hat_func(β::Real, Px_i::Array{Float64,1}, V_p::Array{Float64,2})
    # compute the discounted expected value function for the current type (x_i)
    return β*V_p*Px_i
end

function derivative_func(x::Array{Float64,1}, y::Array{Float64,1}; method::Integer = 0)
    # compute (approximated) first-order derivative with forward finite difference (method = 0), backward finite difference (method = 1).
    x_size = length(x)
    y_size = length(y)
    if x_size == y_size
        dy = zeros(x_size)
        if method == 0
            for x_ind in 1:x_size
                dy[x_ind] = x_ind < x_size ? (y[x_ind+1] - y[x_ind]) / (x[x_ind+1] - x[x_ind]) : dy[x_ind-1]
            end
            return dy
        else
            for x_ind in x_size:-1:1
                dy[x_ind] = x_ind > 1 ? (y[x_ind] - y[x_ind-1]) / (x[x_ind] - x[x_ind-1]) : dy[x_ind+1]
            end
            return dy
        end
    else
        println("WARNING: size mismatch! x_size = $x_size but y_size = $y_size")
        return nothing
    end
end

function rbl_func(V_p::Array{Float64,2}, q_i::Array{Float64,1}, a_grid::Array{Float64,1}; method::Integer = 0)
    # compute risky borrowing limit.

    # (0) check derivative with discretized points (method = 0), , ,
    if method == 0
        dqa = derivative_func(a_grid, q_i .* a_grid)
        dqa_check = Inf
        dqa_iter = length(a_grid)
        while dqa_check > 0
            dqa_check = dqa[dqa_iter]
            dqa_check = dqa[dqa_iter] <= 0 ? break : dqa_iter -= 1
        end
        rbl = a_grid[dqa_iter+1]
        rbl_ind = dqa_iter+1
        a_grid_rbl = a_grid[(dqa_iter+1):end]
        q_i_rbl = q_i[(dqa_iter+1):end]
        V_p_rbl = V_p[(dqa_iter+1):end,:]

    # (1) check derivative with interpolation (method = 1)
    elseif method ==  1 # something wrong neede to be corrected
        q_func = LinearInterpolation(a_grid, q_i, extrapolation_bc = Line())
        dqa_func = LinearInterpolation(a_grid, derivative_func(a_grid, q_i .* a_grid), extrapolation_bc = Line())
        obj_rbl_1(ap) = dqa_func(ap)
        rbl = find_zero(obj_rbl_1, 0)
        rbl_ind = minimum(findall(a_grid .>= rbl))
        a_grid_rbl = cat(rbl, a_grid[rbl_ind:end], dims = 1)
        q_i_rbl = cat(q_func(rbl), q_i[rbl_ind:end], dims = 1)
        x_size = size(V_p,2)
        V_rbl = zeros(1,x_size)
        for x_ind in 1:x_size
            V_func = LinearInterpolation(a_grid, V_p[:,x_ind], extrapolation_bc = Line())
            V_rbl[1,x_ind] = V_func(rbl)
        end
        V_p_rbl = cat(V_rbl, V_p[rbl_ind:end,:], dims = 1)

    # (2) check size with discretized points (method = 2)
    elseif method ==  2
        qa = q_i .* a_grid
        ind_rbl = findall(qa .== minimum(qa))[1]
        rbl = a_grid[ind_rbl]
        rbl_ind = ind_rbl
        a_grid_rbl = a_grid[ind_rbl:end]
        q_i_rbl = q_i[ind_rbl:end]
        V_p_rbl = V_p[ind_rbl:end,:]

    # (3) check size with interpolation (method = 3)
    else
        q_func = LinearInterpolation(a_grid, q_i, extrapolation_bc = Line())
        obj_rbl_3(ap) = ap*q_func(ap)
        results = optimize(obj_rbl_3, a_grid[1], 0)
        rbl = results.minimizer
        rbl_ind = minimum(findall(a_grid .>= rbl))
        a_grid_rbl = cat(rbl, a_grid[rbl_ind:end], dims = 1)
        q_i_rbl = cat(q_func(rbl), q_i[rbl_ind:end], dims = 1)
        x_size = size(V_p,2)
        V_rbl = zeros(1,x_size)
        for x_ind in 1:x_size
            V_func = LinearInterpolation(a_grid, V_p[:,x_ind], extrapolation_bc = Line())
            V_rbl[1,x_ind] = V_func(rbl)
        end
        V_p_rbl = cat(V_rbl, V_p[rbl_ind:end,:], dims = 1)
    end

    # return results
    return rbl, rbl_ind, a_grid_rbl, q_i_rbl, V_p_rbl
end

function ncr_func(dV_hat::Array{Float64,1})
    # compute the non-concave region.
    a_size = length(dV_hat)
    ncr_l, ncr_u = 1, a_size
    # (1) find the lower bound
    V_max, i_max = dV_hat[1], 1
    while i_max < a_size
        if V_max > maximum(dV_hat[(i_max+1):end])
            i_max += 1
            V_max = dV_hat[i_max]
        else
            ncr_l = i_max
            break
        end
    end
    # (2) find the upper bound
    V_min, i_min = dV_hat[end], a_size
    while i_min > 1
        if V_min < minimum(dV_hat[1:(i_min-1)])
            i_min -= 1
            V_min = dV_hat[i_min]
        else
            ncr_u = i_min
            break
        end
    end
    return ncr_l, ncr_u
end

function CoH_func(σ::Real, V_hat::Array{Float64,1}, q_i::Array{Float64,1}, a_grid::Array{Float64,1})
    # compute the cash on hands for current type (x_i).
    return u_func.((derivative_func(a_grid, V_hat) ./ derivative_func(a_grid, q_i .* a_grid)), σ; method = 2) .+ (q_i .* a_grid)
end

function sols_func(β::Real, σ::Real, r_f::Real, earnings::Real, Px_i::Array{Float64,1}, V_p::Array{Float64,2}, a_grid::Array{Float64,1}, q_i::Array{Float64,1}; check_rbl::Integer = 0)

    a_size = length(a_grid)

    # (0) compute the risky borrowing limit and define associated variables
    if check_rbl == 0
        rbl, rbl_ind, a_grid_rbl, q_i_rbl, V_p_rbl = a_grid[1], 1, a_grid, q_i, V_p
    else
        rbl, rbl_ind, a_grid_rbl, q_i_rbl, V_p_rbl = rbl_func(V_p, q_i, a_grid)
    end

    # (1) return global solutions including (a) next-period asset holdings, (b) loan price, (c) cash on hands, (d) associated value functions, (e) associated current asset holdings.
    # contrcut variables above zero asset holding
    a_size_rbl = length(a_grid_rbl)
    V_hat_rbl = V_hat_func(β, Px_i, V_p_rbl)

    # construct the matrix containg all possible "positive" local solutions
    variables.local_sols_bad = zeros(a_size_rbl, 7)
    local_sols[:,1] = a_grid_rbl
    local_sols[:,2] = q_i_rbl
    local_sols[:,3] = V_hat_rbl
    local_sols[:,4] = CoH_func(σ, V_hat_rbl, q_i_rbl, a_grid_rbl)
    local_sols[:,5] = u_func.(local_sols[:,4] .- local_sols[:,2] .* local_sols[:,1], σ) .+ local_sols[:,3]

    # compute the non-concave region
    ncr_l, ncr_u = ncr_func(derivative_func(a_grid_rbl, V_hat_rbl))

    # define the variables whose indices are within the non-concave region
    a_grid_rbl_ncr = a_grid_rbl[ncr_l:ncr_u]
    q_i_rbl_ncr = q_i_rbl[ncr_l:ncr_u]
    V_hat_rbl_ncr = V_hat_rbl[ncr_l:ncr_u]

    # mark global pairs and compute associated current asset position
    for ap_ind in 1:a_size_rbl
        a_temp = local_sols[ap_ind,4] - earnings
        if ap_ind < ncr_l || ap_ind > ncr_u
            local_sols[ap_ind,6] = a_temp / (1+r_f*(a_temp>=0))
            local_sols[ap_ind,7] = 1
        else
            V_temp = u_func.(local_sols[ap_ind,4] .- q_i_rbl_ncr .* a_grid_rbl_ncr, σ) .+ V_hat_rbl_ncr
            ind_max_V_temp = findall(V_temp .== maximum(V_temp))[1]
            if (ap_ind - ncr_l + 1) == ind_max_V_temp
                local_sols[ap_ind,6] = a_temp / (1+r_f*(a_temp>=0))
                local_sols[ap_ind,7] = 1
            end
        end
    end

    # export "positive" global solutions
    global_sols = local_sols[local_sols[:,7] .== 1.0, 1:6]

    # if holding zero asset holding is NOT included, make it included!
    if global_sols[1,1] != a_grid_rbl[1]
        # locate the first point in the global solutions
        ind_a1 = findall(a_grid_rbl .== global_sols[1,1])[1]
        # define the objective function as in equation (21) in Fella's paper
        obj_zero(x) = u_func(x-q_i_rbl[1]*a_grid_rbl[1], σ) + V_hat_rbl[1] - u_func(x-q_i_rbl[ind_a1]*a_grid_rbl[ind_a1], σ) - V_hat_rbl[ind_a1]
        C1 = find_zero(obj_zero, (q_i_rbl[1]*a_grid_rbl[1], global_sols[1,4]))
        a1 = (C1-earnings) / (1+r_f*(C1-earnings>0))
        V1 = u_func(C1-q_i_rbl[1]*a_grid_rbl[1], σ) + V_hat_rbl[1]
        # expand the original global pairs
        global_sols = cat([a_grid_rbl[1] q_i_rbl[1] V_hat_rbl[1] C1 V1 a1], global_sols, dims = 1)
    end

    # (2) update value and policy functions.
    # construct containers
    V = zeros(a_size)
    policy_a = zeros(a_size)

    if check_rbl == 0
        # define interpolated functions
        V_func = LinearInterpolation(global_sols[:,6], global_sols[:,5], extrapolation_bc = Line())
        a_func = LinearInterpolation(global_sols[:,6], global_sols[:,1], extrapolation_bc = Line())

        # extrapolate value and policy functions
        for a_ind in 1:a_size_rbl
            if a_grid_rbl[a_ind] >= global_sols[1,6]
                V[a_ind] = V_func(a_grid_rbl[a_ind])
                policy_a[a_ind] = a_func(a_grid_rbl[a_ind])
            else
                # compute the value as in equation (20) in Fella's paper
                V[a_ind] = u_func(earnings+(1+r_f*(a_grid_rbl[a_ind]>=0))*a_grid_rbl[a_ind]-q_i_rbl[1]*a_grid_rbl[1], σ) + V_hat_rbl[1]
                policy_a[a_ind] = a_grid_rbl[1]
            end
        end
    else
        # locate the first non-positive element
        ind_ap_nonpos = maximum(findall(global_sols[:,1] .<= 0))

        # define interpolated functions above a' = 0
        V_func = LinearInterpolation(global_sols[:,6], global_sols[:,5], extrapolation_bc = Line())
        a_func = LinearInterpolation(global_sols[:,6], global_sols[:,1], extrapolation_bc = Line())

        # define interpolated functions below a' = 0
        # qa_rbl_func = LinearInterpolation(global_sols_rbl[:,1], (global_sols_rbl[:,2].*global_sols_rbl[:,1]), extrapolation_bc = Line())
        # V_hat_rbl_func = LinearInterpolation(global_sols_rbl[:,1], global_sols_rbl[:,3], extrapolation_bc = Line())

        # extrapolate value and policy functions
        for a_ind in 1:a_size
            if a_grid[a_ind] >= global_sols[ind_ap_nonpos,6]
                V[a_ind] = V_func(a_grid[a_ind])
                policy_a[a_ind] = a_func(a_grid[a_ind])
            else
                # compute the value as in equation (20) in Jang and Lee's paper
                V_temp = u_func.(earnings .+ (1+r_f*(a_grid[a_ind]>=0))*a_grid[a_ind] .- global_sols[1:ind_ap_nonpos,2].*global_sols[1:ind_ap_nonpos,1], σ) .+ global_sols[1:ind_ap_nonpos,3]
                a_ind_temp = findall(V_temp .== maximum(V_temp))[1]
                V[a_ind] = V_temp[a_ind_temp]
                policy_a[a_ind] = a_grid_rbl[a_ind_temp]
            end
        end
    end

    # return results
    return global_sols, V, policy_a
end

function price!(variables::mut_vars, parameters::NamedTuple)
    # update the price schedule.
    @unpack ξ, r_f, a_grid_neg, a_size_neg, Px, x_grid, x_size = parameters
    α = 0.15
    for x_ind in 1:x_size
        for ap_ind in 1:a_size_neg
            revenue = 0
            for xp_ind in 1:x_size
                pp_i, νp_i = x_grid[xp_ind,:]
                earnings = pp_i
                if variables.V_good_default[ap_ind,xp_ind] >= variables.V_good_repay[ap_ind,xp_ind]
                    revenue += Px[x_ind,xp_ind]*ξ*earnings
                else
                    revenue += Px[x_ind,xp_ind]*(-a_grid_neg[ap_ind])
                end
            end
            q_update = α*(revenue / ((1+r_f)*(-a_grid_neg[ap_ind]))) + (1-α)*variables.q[ap_ind,x_ind]
            variables.q[ap_ind,x_ind] = q_update < (1/(1+r_f)) ? q_update : 1/(1+r_f)
        end
    end
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
    V_good_p = similar(variables.V_good)
    V_bad_p = similar(variables.V_bad)

    while crit > tol && iter < iter_max

        println("iter = $iter")
        # copy the current value functions to the pre-specified containers
        copyto!(V_good_p, variables.V_good)
        copyto!(V_bad_p, variables.V_bad)

        # start looping over each household's type
        for x_ind in 1:x_size

            # unpack or construct the individual states and variables
            p_i, ν_i = x_grid[x_ind,:]
            earnings = p_i
            β_adj = ν_i*β
            q_i = variables.q[:,x_ind]
            Px_i = Px[x_ind,:]

            #-------------------------------------------------------------#
            # compute global solutions, update value and policy functions #
            #-------------------------------------------------------------#
            # (1) bad credit history
            println("x_ind = $x_ind, bad")
            V_good_p_pos = V_good_p[ind_a_zero:end,:]
            q_i_pos = q_i[ind_a_zero:end]
            global_sols_B, variables.V_bad[:,x_ind], variables.policy_a_bad[:,x_ind] = sols_func(β_adj, σ, r_f, earnings, Px_i, λ*V_good_p_pos .+ (1-λ)*V_bad_p, a_grid_pos, q_i_pos)

            # (2) good credit history with repayment
            println("x_ind = $x_ind, good")
            global_sols_G, variables.V_good_repay[:,x_ind], variables.policy_a_good_repay[:,x_ind] = sols_func(β_adj, σ, r_f, earnings, Px_i, V_good_p, a_grid, q_i; check_rbl = 1)

            # (3) good credit history with defaulting
            V_hat_bad = V_hat_func(β_adj, Px_i, V_bad_p)
            variables.V_good_default[:,x_ind] .= u_func(earnings*(1-ξ), σ) + V_hat_bad[1]

            # (4) good credit history
            variables.V_good[:,x_ind] = max.(variables.V_good_repay[:,x_ind], variables.V_good_default[:,x_ind])
        end

        # update price
        price!(variables, parameters)

        # check convergence
        crit = max(norm(variables.V_good-V_good_p, Inf), norm(variables.V_bad-V_bad_p, Inf))

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end
