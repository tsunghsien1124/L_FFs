function fixed_matrix_initial_good(V_good::Array{Float64,2}, U_good_repay::Array{Float64,2}, ν::Array{Float64,2}, β::Real, Px::Array{Float64,2}; tol = 1E-8, iter_max = 10000)
    # compute the initla guess for the value function of good credit history
    iter = 0
    crit = Inf
    V_good_p = similar(V_good)
    while crit > tol && iter < iter_max
        copyto!(V_good_p, V_good)
        V_good = U_good_repay + (ν*β) .* (V_good_p*transpose(Px))
        crit = norm(V_good-V_good_p, Inf)
        iter += 1
    end
    return V_good
end

function fixed_matrix_bad(V_bad::Array{Float64,2}, V_good_pos::Array{Float64,2}, U_bad::Array{Float64,2}, ν_pos::Array{Float64,2}, β::Real, λ::Real, Px::Array{Float64,2}; tol = 1E-8, iter_max = 10000)
    # compute the initla guess for the value function of good credit history
    iter = 0
    crit = Inf
    V_bad_p = similar(V_bad)
    while crit > tol && iter < iter_max
        copyto!(V_bad_p, V_bad)
        V_bad = U_bad + (ν_pos*β*λ) .* (V_good_pos*transpose(Px)) + (ν_pos*β*(1-λ)) .* (V_bad_p*transpose(Px))
        crit = norm(V_bad-V_bad_p, Inf)
        iter += 1
    end
    return V_bad
end

function initial_guess(q::Array{Float64,2}, parameters::NamedTuple; tol = 1E-8, iter_max = 10000)
    # compute initial guess of value functions

    # unpack parameters
    @unpack a_grid, a_size, a_size_pos, a_size_neg, ind_a_zero, p_size, ν_grid, ν_size, Px, x_grid, x_size, r_f, β, σ, ξ, λ = parameters

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf

    # construct containers
    V_good = zeros(a_size, x_size)
    V_good_repay = zeros(a_size, x_size)
    V_good_default = zeros(a_size, x_size)
    V_bad = zeros(a_size_pos, x_size)

    V_good_p = similar(V_good)
    V_good_repay_p = similar(V_good_repay)
    V_good_default_p = similar(V_good_default)
    V_bad_p = similar(V_bad)

    U_good_repay = zeros(a_size, x_size)
    U_good_default = zeros(a_size, x_size)
    U_bad = zeros(a_size_pos, x_size)

    # define utility functions
    for a_ind in 1:a_size, x_ind in 1:x_size
        p_i = x_grid[x_ind, 1]
        U_good_repay[a_ind,x_ind] = a_grid[a_ind] >= 0 ? u_func(p_i+r_f*a_grid[a_ind], σ) : u_func(p_i+a_grid[a_ind]-q[a_ind,x_ind]*a_grid[a_ind], σ)
        U_good_default[a_ind,x_ind] = u_func(p_i*(1-ξ), σ)
    end
    U_bad = U_good_repay[ind_a_zero:end,:]

    # define some useful matrices
    ν = kron(transpose(ν_grid), ones(a_size, p_size))
    ν_pos = ν[1:a_size_pos,:]

    # initialize the value function of good credit history
    V_good = fixed_matrix_initial_good(V_good, U_good_repay, ν, β, Px)

    while crit > tol && iter < iter_max

        # copy the current value functions to the pre-specified containers
        copyto!(V_good_p, V_good)
        copyto!(V_good_repay_p, V_good_repay)
        copyto!(V_good_default_p, V_good_default)
        copyto!(V_bad_p, V_bad)

        # update value functions
        V_bad = fixed_matrix_bad(V_bad_p, V_good_p[ind_a_zero:end,:], U_bad, ν_pos, β, λ, Px)
        V_good_repay = U_good_repay + (ν*β) .* (V_good_p*transpose(Px))
        V_good_default = U_good_default + (ν*β) .* kron(transpose(V_bad[1,:])*transpose(Px), ones(a_size,1))
        V_good = max.(V_good_repay, V_good_default)

        # check convergence
        crit = max(norm(V_good-V_good_p, Inf), norm(V_bad-V_bad_p, Inf))

        # update the iteration number
        iter += 1
    end

    # return outputs
    return V_good, V_good_repay, V_good_default, V_bad
end

# define pricing function and default probability
q = ones(a_size, x_size)
for x_ind in 1:x_size
    p, ν = x_grid[x_ind,:]
    for a_ind in 1:a_size_neg
        q[a_ind,x_ind] = (p/x_grid[end,1]) * (a_grid[a_ind]/a_grid[1]) / (1 + r_f)
    end
end

function sols_func(β::Real, σ::Real, r_f::Real, earnings::Real, Px_i::Array{Float64,1}, V_p::Array{Float64,2}, a_grid::Array{Float64,1}, q_i::Array{Float64,1}; check_rbl = 0)
    # (1) return global solutions including (a) next-period asset holdings, (b) loan price, (c) cash on hands, (d) associated value functions, (e) associated current asset holdings.

    # compute the risky borrowing limit and define associated variables
    if check_rbl == 1
        rbl, rbl_ind, a_grid_rbl, q_i_rbl, V_p_rbl = rbl_func(V_p, a_grid, q_i)
    else
        rbl, rbl_ind, a_grid_rbl, q_i_rbl, V_p_rbl = a_grid[1], 1, a_grid, q_i, V_p
    end

    # construct the matrix containg all possible local solutions
    a_size_rbl = length(a_grid_rbl)
    local_sols = zeros(a_size_rbl, 7)
    V_hat_rbl = V_hat_func(β, Px_i, V_p_rbl)
    local_sols[:,1] = a_grid_rbl
    local_sols[:,2] = q_i_rbl
    local_sols[:,3] = V_hat_rbl
    local_sols[:,4] = CoH_func(β, σ, Px_i, V_p_rbl, a_grid_rbl, q_i_rbl)
    local_sols[:,5] = u_func.(local_sols[:,4] .- local_sols[:,2] .* local_sols[:,1], σ) .+ local_sols[:,3]

    # compute the non-concave region
    ncr_l, ncr_u = ncr_func(β, Px_i, V_p_rbl, a_grid_rbl)

    # define the variables whose indices are within the non-concave region
    a_grid_ncr = a_grid_rbl[ncr_l:ncr_u]
    q_i_ncr = q_i_rbl[ncr_l:ncr_u]
    V_hat_ncr = V_hat_rbl[ncr_l:ncr_u]

    # mark global pairs and compute associated current asset position
    for ap_ind in 1:a_size_rbl
        a_temp = local_sols[ap_ind,4] - earnings
        if ap_ind < ncr_l || ap_ind > ncr_u
            local_sols[ap_ind,6] = a_temp / (1+r_f*(a_temp>=0))
            local_sols[ap_ind,7] = 1
        else
            V_temp = u_func.(local_sols[ap_ind,4] .- q_i_ncr .* a_grid_ncr, σ) .+ V_hat_ncr
            ind_max_V_temp = findall(V_temp .== maximum(V_temp))[1]
            if (ap_ind - ncr_l + 1) == ind_max_V_temp
                local_sols[ap_ind,6] = a_temp / (1+r_f*(a_temp>=0))
                local_sols[ap_ind,7] = 1
            end
        end
    end

    # export global solutions
    global_sols = local_sols[local_sols[:,7] .== 1.0, 1:6]

    # if the borrowing limit is NOT included, make it included!
    if global_sols[1,1] != rbl
        # locate the first point in the global solutions
        a1_ind = findall(a_grid_rbl .== global_sols[1,1])[1]
        # define the objective function as in equation (21) in Fella's paper
        # obj(x) = u_func(earnings+(1+r_f*(x>0))*x-q_i_rbl[1]*a_grid_rbl[1], σ) + V_hat_rbl[1] - u_func(earnings+(1+r_f*(x>0))*x-q_i_rbl[a1_ind]*a_grid_rbl[a1_ind], σ) - V_hat_rbl[a1_ind]
        obj(x) = u_func(x-q_i_rbl[1]*a_grid_rbl[1], σ) + V_hat_rbl[1] - u_func(x-q_i_rbl[a1_ind]*a_grid_rbl[a1_ind], σ) - V_hat_rbl[a1_ind]
        C1 = find_zero(obj, (q_i_rbl[a1_ind]*a_grid_rbl[a1_ind], global_sols[1,4]))
        a1 = (C1-earnings) / (1+r_f*(C1-earnings>0))
        V1 = u_func(C1-q_i_rbl[1]*a_grid_rbl[1], σ) + V_hat_rbl[1]
        # expand the original global pairs
        global_sols = cat([a_grid_rbl[1] q_i_rbl[1] V_hat_rbl[1] C1 V1 a1], global_sols, dims = 1)
    end

    # (2) update value and policy functions.
    a_size = length(a_grid)
    V = zeros(a_size, 1)
    policy_a = zeros(a_size, 1)

    if check_rbl == 0
        # define interpolated functions
        V_func = LinearInterpolation(global_sols[:,6], global_sols[:,5], extrapolation_bc = Line())
        a_func = LinearInterpolation(global_sols[:,6], global_sols[:,1], extrapolation_bc = Line())

        # extrapolate value and policy functions
        for a_ind in 1:a_size
            if a_grid[a_ind] >= global_sols[1,6]
                V[a_ind] = V_func(a_grid[a_ind])
                policy_a[a_ind] = a_func(a_grid[a_ind])
            else
                # compute the value as in equation (20) in Fella's paper
                V[a_ind] = u_func(earnings+(1+r_f*(a_grid[a_ind]>=0))*a_grid[a_ind]-q_i_rbl[1]*a_grid_rbl[1], σ) + V_hat_rbl[1]
                policy_a[a_ind] = a_grid_rbl[1]
            end
        end
    else
        # locate the point such that a' = 0
        ind_ap_zero = findall(global_sols[:,1] .== 0)[1]
        a_ap_zero = global_sols[ind_ap_zero,6]

        # define interpolated functions above a' = 0
        V_func = LinearInterpolation(global_sols[ind_ap_zero:end,6], global_sols[ind_ap_zero:end,5], extrapolation_bc = Line())
        a_func = LinearInterpolation(global_sols[ind_ap_zero:end,6], global_sols[ind_ap_zero:end,1], extrapolation_bc = Line())

        # extrapolate value and policy functions
        for a_ind in 1:a_size
            if a_grid[a_ind] >= a_ap_zero
                V[a_ind] = V_func(a_grid[a_ind])
                policy_a[a_ind] = a_func(a_grid[a_ind])
            else
                # compute the value as in equation (20) in Jang and Lee's paper
                c_temp = earnings .+ (1+r_f*(a_grid[a_ind]>=0))*a_grid[a_ind] .- global_sols[1:(ind_ap_zero-1),2].*global_sols[1:(ind_ap_zero-1),1]
                V_temp = u_func.(c_temp, σ) .+ global_sols[1:(ind_ap_zero-1),3]
                ind_ap_temp = findall(V_temp .== maximum(V_temp))[1]
                V[a_ind] = V_temp[ind_ap_temp]
                policy_a[a_ind] = global_sols[ind_ap_temp,1]
            end
        end
    end

    # return results
    return global_sols, V, policy_a
end
