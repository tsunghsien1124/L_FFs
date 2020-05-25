function para_func(; λ_H::Real = 0.10,       # history rased probability
                     λ_B::Real = 0.80,       # bank's survival rate
                     β::Real = 0.96,         # discount factor
                     ξ::Real = 0.30,         # garnishment rate
                     σ::Real = 2,            # CRRA coefficient
                     L::Real = 10,           # targeted leverage ratio
                     r_bf::Real = 0.01,      # targeted excess return
                     r_f::Real = 0.03,       # risk-free rate
                     ρ_p::Real = 0.95,       # AR(1) of persistent shock
                     σ_p::Real = 0.10,       # s.d. of persistent shock
                     p_size::Integer = 15,    # no. of persistent shock
                     ν_size::Integer = 2,    # no. of preference shock
                     ν::Real = 0.70,         # level of patience
                     pν::Real = 0.05,        # probability of patience
                     a_min::Real = -1,       # min of asset holding
                     a_max::Real = 50,       # max of asset holding
                     a_size::Integer = 200)  # number of the grid asset holding
    #------------------------------------------------------#
    # contruct an immutable object containg all paramters. #
    #------------------------------------------------------#

    # persistent shock
    Mp = rouwenhorst(p_size, ρ_p, σ_p)
    Pp = Mp.p
    p_grid = exp.(collect(Mp.state_values))

    # preference schock
    ν_grid = [ν, 1]
    Pν = repeat([pν 1-pν], ν_size, 1)

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

    # solve the steady state of ω and θ to match targeted parameters
    ω, θ = solve_ss_func_no_r_bf(β, λ_B, L, r_bf, r_f)

    # return values
    return (λ_H = λ_H, λ_B = λ_B, β = β, ξ = ξ, σ = σ, L = L, r_bf = r_bf, r_f = r_f, ω = ω, θ = θ, a_grid = a_grid, ind_a_zero = ind_a_zero, a_size = a_size, a_size_pos = a_size_pos, a_size_neg = a_size_neg, a_grid_neg = a_grid_neg, a_grid_pos = a_grid_pos, Pp = Pp, p_grid = p_grid, p_size = p_size, Pν = Pν, ν_grid = ν_grid, ν_size = ν_size, Px = Px, x_grid = x_grid, x_size = x_size, x_ind = x_ind)
end

function solve_ss_func_no_r_bf(β::Real, λ_B::Real, L::Real, r_bf::Real, r_f::Real)
    #-----------------------#
    # compute steady state. #
    #-----------------------#
    r_b = r_f + r_bf
    G_No_N = λ_B*((r_b-r_f)*L+(1+r_f))
    G_Nn_ωN = (1-λ_B)*(1+r_b)*L
    ω = (1-G_No_N) / G_Nn_ωN
    G_Nn_N = ω*G_Nn_ωN
    θ = ((β*(1-λ_B))/(λ_B*L))*(G_No_N/(1-β*G_No_N))
    return ω, θ
end

function solve_ss_func(β::Real, λ_B::Real, L::Real, r_bf::Real, r_f::Real)
    #-----------------------#
    # compute steady state. #
    #-----------------------#
    check = Inf
    check_iter = 0
    ω = 0
    θ = 0
    while check > 0
        r_b = r_f + r_bf
        G_No_N = λ_B*((r_b-r_f)*L+(1+r_f))
        G_Nn_ωN = (1-λ_B)*(1+r_b)*L
        ω = (1-G_No_N) / G_Nn_ωN
        G_Nn_N = ω*G_Nn_ωN
        θ = ((β*(1-λ_B))/(λ_B*L))*(G_No_N/(1-β*G_No_N))
        ϕ = L*θ
        Λ = β*(1-λ_B+λ_B*ϕ)
        γ = (Λ*(r_b-r_f))/θ
        r_bf1 = (1-λ_B*(1+r_f)*(1-L)) / ((λ_B+ω*(1-λ_B))*L) - 1 - r_f
        L1 = (Λ*(1+ω*(1-λ_B)*(1+r_f))) / (λ_B*θ + ω*(1-λ_B)*(θ+Λ*(1+r_f)))
        check = max(r_bf1 - r_bf, L1 - L)
        r_bf = r_bf1
        L = L1
        check_iter += 1
    end
    return ω, θ, r_bf
end

mutable struct mut_vars
    q::Array{Float64,3}
    V_good::Array{Float64,3}
    V_bad::Array{Float64,2}
    policy_a_good::Array{Float64,3}
    policy_a_bad::Array{Float64,2}
    policy_a_good_repay_matrix::SparseMatrixCSC{Float64,Int64}
    policy_a_good_default_matrix::SparseMatrixCSC{Float64,Int64}
    policy_a_bad_matrix::SparseMatrixCSC{Float64,Int64}
    μ::Array{Float64,1}
    Pμ::SparseMatrixCSC{Float64,Int64}
    A::Array{Float64,1}
end

function vars_func(parameters::NamedTuple)
    #-------------------------------------------------------------#
    # construct a mutable object containing endogenous variables. #
    #-------------------------------------------------------------#

    # unpack parameters
    @unpack β, σ, r_f, r_bf, a_grid, a_grid_neg, a_grid_pos, a_size, a_size_neg, a_size_pos, ind_a_zero, x_grid, x_size, p_grid = parameters

    # define pricing related variables
    # (1: price, 2: price derivative, 3: size, 4: size derivative)
    q = zeros(a_size, x_size, 4)
    q_mode = 0
    if q_mode == 0
        q[:,:,1] .= 1.0
        q[1:ind_a_zero,:,1] .= 1/(1+r_f+r_bf)
    else
        q[:,:,1] .= 1.0
        for x_ind in 1:x_size
            p_i, ν_i = x_grid[x_ind,:]
            for a_ind in 1:ind_a_zero
                q[a_ind,x_ind,1] = (p_i/p_grid[end]) * ((a_grid[1]-a_grid[a_ind])/a_grid[1]) / (1 + r_f + r_bf)
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

    # define policy matrices
    policy_a_good_repay_matrix = spzeros(x_size*a_size, a_size)
    policy_a_good_default_matrix = spzeros(x_size*a_size, a_size_pos)
    policy_a_bad_matrix = spzeros(x_size*a_size_pos, a_size_pos)

    # define the type distribution and its transition matrix
    μ_size_good = x_size*a_size
    μ_size_bad = x_size*a_size_pos
    μ_size = μ_size_good + μ_size_bad
    μ = zeros(μ_size)
    Pμ = spzeros(μ_size, μ_size)

    # define aggregate objects
    A = zeros(4)

    # return outputs
    variables = mut_vars(q, V_good, V_bad, policy_a_good, policy_a_bad, policy_a_good_repay_matrix, policy_a_good_default_matrix, policy_a_bad_matrix, μ, Pμ, A)
    return variables
end

mutable struct mut_vars_MIT
    q::Array{Float64,4}
    V_good::Array{Float64,4}
    V_bad::Array{Float64,3}
    policy_a_good::Array{Float64,4}
    policy_a_bad::Array{Float64,3}
    policy_a_good_repay_matrix::Array{SparseMatrixCSC{Float64,Int64},1}
    policy_a_good_default_matrix::Array{SparseMatrixCSC{Float64,Int64},1}
    policy_a_bad_matrix::Array{SparseMatrixCSC{Float64,Int64},1}
    μ::Array{Float64,2}
    Pμ::Array{SparseMatrixCSC{Float64,Int64},1}
    A::Array{Float64,2}
    P::Array{Float64,2}
end

function vars_MIT_func(z_guess::Array{Float64,1}, N_guess::Array{Float64,1}, variables::mut_vars, parameters::NamedTuple; T::Integer = 150)
    #------------------------------------------------------------------------#
    # construct a mutable object containing endo. variables for a MIT shock. #
    #------------------------------------------------------------------------#

    # unpack parameters
    @unpack β, σ, λ_B, r_f, r_bf, L, ω, θ, a_grid, a_grid_neg, a_grid_pos, a_size, a_size_neg, a_size_pos, ind_a_zero, x_grid, x_size, p_grid = parameters

    # define pricing related variables
    # (1: price, 2: price derivative, 3: size, 4: size derivative)
    q = ones(a_size, x_size, 4, T)
    q[:,:,:,end] .= variables.q

    # define value functions (1: good, 2: good and repay, 3: good but default)
    V_good = zeros(a_size, x_size, 3, T)
    V_good[:,:,:,end] .= variables.V_good
    V_bad = zeros(a_size_pos, x_size, T)
    V_bad[:,:,end] .= variables.V_bad

    # define policy functions
    policy_a_good = zeros(a_size, x_size, 3, T)
    policy_a_good[:,:,:,end] .= variables.policy_a_good
    policy_a_bad = zeros(a_size_pos, x_size, T)
    policy_a_bad[:,:,end] .= variables.policy_a_bad

    # define policy matrices
    policy_a_good_repay_matrix = [spzeros(x_size*a_size, a_size) for t in 1:T]
    policy_a_good_repay_matrix[end] .= variables.policy_a_good_repay_matrix
    policy_a_good_default_matrix = [spzeros(x_size*a_size, a_size_pos) for t in 1:T]
    policy_a_good_default_matrix[end] .= variables.policy_a_good_default_matrix
    policy_a_bad_matrix = [spzeros(x_size*a_size_pos, a_size_pos) for t in 1:T]
    policy_a_bad_matrix[end] .= variables.policy_a_bad_matrix

    # define the type distribution and its transition matrix
    μ_size_good = x_size*a_size
    μ_size_bad = x_size*a_size_pos
    μ_size = μ_size_good + μ_size_bad
    μ = zeros(μ_size,T)
    μ[:,end] .= variables.μ
    Pμ = [spzeros(μ_size,μ_size) for t in 1:T]
    Pμ[end] .= variables.Pμ

    # define aggregate objects: QB[1], D[2], N[3], L[4]
    A = zeros(4,T)
    A[:,end] .= variables.A
    A[3,:] .= N_guess
    # A[4,:] .= variables.A[4]

    # compute aggregate prices: z[1], G_N[2], φ[3], Λ[4], r_b[5], γ[6]
    P = zeros(6,T)
    P[1,:] = z_guess
    P[2,:] = N_guess ./ [A[3,end]; N_guess[1:(end-1)]]
    # P[5,:] .= parameters.r_f + parameters.r_bf
    for t in T:-1:1
        if t == T
            P[3,t] = θ*A[4,t]
            P[4,t] = β*(1-λ_B+λ_B*P[3,t])
            P[5,t] = (P[2,t]-λ_B*(1+r_f)*(1-A[4,t])) / ((λ_B+ω*(1-λ_B))*A[4,t]) - 1
            P[6,t] = P[4,t]*(P[5,t]-r_f)/θ
        else
            P[3,t] = θ*A[4,t+1]
            P[4,t] = β*(1-λ_B+λ_B*P[3,t])
            A[4,t] = (P[4,t]*(P[2,t]+ω*(1-λ_B)*(1+r_f))) / (λ_B*θ + ω*(1-λ_B)*(θ+P[4,t]*(1+r_f)))
            P[5,t] = (P[2,t]-λ_B*(1+r_f)*(1-A[4,t])) / ((λ_B+ω*(1-λ_B))*A[4,t]) - 1
            P[6,t] = P[4,t+1]*(P[5,t]-r_f)/θ
        end
    end

    # return outputs
    variables_MIT = mut_vars_MIT(q, V_good, V_bad, policy_a_good, policy_a_bad, policy_a_good_repay_matrix, policy_a_good_default_matrix, policy_a_bad_matrix, μ, Pμ, A, P)
    return variables_MIT
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

function rbl_func(V_p::Array{Float64,2}, q_i::Array{Float64,2}, a_grid::Array{Float64,1}; method::Integer = 0)
    #--------------------------------#
    # compute risky borrowing limit. #
    #--------------------------------#

    # (0) check derivative with discretized points (method = 0)
    if method == 0
        Dqa_check = Inf
        Dqa_iter = length(a_grid)
        while Dqa_check > 0
            Dqa_check = q_i[Dqa_iter, 4]
            Dqa_check = q_i[Dqa_iter, 4] <= 0 ? break : Dqa_iter -= 1
        end
        # rbl_ind = Dqa_iter == length(a_grid) ? Dqa_iter : Dqa_iter + 1
        rbl_ind = Dqa_iter + 1
        rbl = a_grid[rbl_ind]
        V_p_rbl = V_p[rbl_ind:end,:]
        q_i_rbl = q_i[rbl_ind:end,:]
        a_grid_rbl = a_grid[rbl_ind:end]
        a_size_rbl = length(a_grid_rbl)

    # (1) check size with discretized points (method = 1)
    elseif method ==  1
        rbl_ind = findall(q_i[:,3] .== minimum(q_i[:,3]))[1]
        rbl = a_grid[rbl_ind]
        V_p_rbl = V_p[rbl_ind:end,:]
        q_i_rbl = q_i[rbl_ind:end,:]
        a_grid_rbl = a_grid[rbl_ind:end]
        a_size_rbl = length(a_grid_rbl)

    # (2) check size with interpolation (method = 2)
    else
        q_func = LinearInterpolation(a_grid, q_i[:,1], extrapolation_bc = Line())
        qa_func = LinearInterpolation(a_grid, q_i[:,3], extrapolation_bc = Line())
        obj_rbl_qaitp(ap) = qa_func(ap)
        results = optimize(obj_rbl_qaitp, a_grid[1], 0)
        rbl = results.minimizer
        rbl_ind = minimum(findall(a_grid .>= rbl))
        V_rbl = zeros(1,size(V_p,2))
        for x_ind in 1:size(V_p,2)
            V_func = LinearInterpolation(a_grid, V_p[:,x_ind], extrapolation_bc = Line())
            V_rbl[1,x_ind] = V_func(rbl)
        end
        V_p_rbl = cat(V_rbl, V_p[rbl_ind:end,:], dims = 1)
        q_i_rbl = cat([q_func(rbl) 0 qa_func(rbl) 0], q_i[rbl_ind:end,:], dims = 1)
        a_grid_rbl = cat(rbl, a_grid[rbl_ind:end], dims = 1)
        a_size_rbl = length(a_grid_rbl)
    end

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
        V_0 = u_func(C_0-q_i_pos[1,3], σ) + V_hat_pos[1]
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
    return V, policy_a
end

function convex_func(ap::Array{Float64,1}, a_grid::Array{Float64,1}; matrix_display::Integer = 0)
    #------------------------------#
    # find the convex combination. #
    #------------------------------#
    ap_size = length(ap)
    if ap_size == 1
        ap_i = ap[1]
        ind_lower = maximum(findall(a_grid .<= ap_i))
        ind_upper = minimum(findall(a_grid .>= ap_i))
        if ind_lower == ind_upper
            coef_lower = 1.0
            coef_upper = 1.0
        else
            coef_lower = (a_grid[ind_upper]-ap_i) / (a_grid[ind_upper]-a_grid[ind_lower])
            coef_upper = (ap_i-a_grid[ind_lower]) / (a_grid[ind_upper]-a_grid[ind_lower])
        end
        if matrix_display == 0
            return (ind_lower, ind_upper, coef_lower, coef_upper)
        else
            a_size = length(a_grid)
            matrix_results = spzeros(1,a_size)
            matrix_results[1,ind_lower] = coef_lower
            matrix_results[1,ind_upper] = coef_upper
            return matrix_results
        end
    else
        results = zeros(ap_size,4)
        for ap_ind in 1:ap_size
            ap_i = ap[ap_ind]
            ind_lower = maximum(findall(a_grid .<= ap_i))
            ind_upper = minimum(findall(a_grid .>= ap_i))
            if ind_lower == ind_upper
                coef_lower = 1.0
                coef_upper = 1.0
            else
                coef_lower = (a_grid[ind_upper]-ap_i) / (a_grid[ind_upper]-a_grid[ind_lower])
                coef_upper = (ap_i-a_grid[ind_lower]) / (a_grid[ind_upper]-a_grid[ind_lower])
            end
            results[ap_ind,:] = [ind_lower, ind_upper, coef_lower, coef_upper]
        end
        if matrix_display == 0
            return results
        else
            a_size = length(a_grid)
            matrix_results = spzeros(ap_size,a_size)
            for ap_ind in 1:ap_size
                matrix_results[ap_ind,convert(Int,results[ap_ind,1])] = results[ap_ind,3]
                matrix_results[ap_ind,convert(Int,results[ap_ind,2])] = results[ap_ind,4]
            end
            return matrix_results
        end
    end
end

function policy_matrix_func(policy_a::Array{Float64,2}, a_grid::Array{Float64,1}; V_good::Array{Float64,3} = zeros(1,1,1))
    #---------------------------------------------------------#
    # construct the matrix representaion of policy functions. #
    #---------------------------------------------------------#
    a_size = length(a_grid)
    ind_a_zero = findall(a_grid .>= 0)[1]
    a_grid_pos = a_grid[ind_a_zero:end]
    a_size_pos = length(a_grid_pos)
    a_policy_size, x_size = size(policy_a)
    if a_size == a_policy_size
        policy_a_repay_matrix = spzeros(x_size*a_size, a_size)
        policy_a_default_matrix = spzeros(x_size*a_size, a_size_pos)
        for x_ind in 1:x_size
            ind_repay = V_good[:,x_ind,2] .> V_good[:,x_ind,3]
            ind_default = V_good[:,x_ind,3] .>= V_good[:,x_ind,2]
            results_repay = convex_func(policy_a[:,x_ind], a_grid; matrix_display = 1)
            results_repay[ind_default,:] .= 0
            results_repay = dropzeros(results_repay)
            results_default = convex_func(policy_a[:,x_ind], a_grid; matrix_display = 1)
            results_default[ind_repay,:] .= 0
            results_default = dropzeros(results_default)
            ind_r1 = 1 + a_size*(x_ind-1)
            ind_r2 = a_size*x_ind
            policy_a_repay_matrix[ind_r1:ind_r2,:] = results_repay
            policy_a_default_matrix[ind_r1:ind_r2,:] = results_default[:,ind_a_zero:end]
        end
        return policy_a_repay_matrix, policy_a_default_matrix
    else
        policy_a_matrix = spzeros(x_size*a_size_pos, a_size_pos)
        for x_ind in 1:x_size
            ind_r1 = 1 + a_size_pos*(x_ind-1)
            ind_r2 = a_size_pos*x_ind
            policy_a_matrix[ind_r1:ind_r2,:] = convex_func(policy_a[:,x_ind], a_grid_pos; matrix_display = 1)
        end
        return policy_a_matrix
    end
end

function price_func!(variables::mut_vars, parameters::NamedTuple)
    #-------------------------------------------------------#
    # update the price schedule and associated derivatives. #
    #-------------------------------------------------------#
    @unpack ξ, r_bf, r_f, a_grid, a_grid_neg, a_size_neg, ind_a_zero, Px, x_grid, x_size = parameters
    α = 1    # parameter controling update speed
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
                q_update = α*(revenue / ((1+r_f+r_bf)*(-a_grid_neg[ap_ind]))) + (1-α)*variables.q[ap_ind,x_ind,1]
                variables.q[ap_ind,x_ind,1] = q_update < (1/(1+r_f+r_bf)) ? q_update : 1/(1+r_f+r_bf)
            else
                variables.q[ap_ind,x_ind,1] = 1/(1+r_f+r_bf)
            end
        end
    end
    variables.q[:,:,2] = derivative_func(a_grid, variables.q[:,:,1])
    variables.q[:,:,3] = variables.q[:,:,1] .* repeat(a_grid, 1, x_size)
    variables.q[:,:,4] = derivative_func(a_grid, variables.q[:,:,3])
end

function price_MIT_func!(t::Integer, variables_MIT::mut_vars_MIT, parameters::NamedTuple)
    #-------------------------------------------------------#
    # update the price schedule and associated derivatives. #
    #-------------------------------------------------------#
    @unpack ξ, θ, r_f, a_grid, a_grid_neg, a_size_neg, ind_a_zero, Px, x_grid, x_size = parameters
    α = 1    # parameter controling update speed
    r_b = variables_MIT.P[5,t+1]
    for x_ind in 1:x_size
        for ap_ind in 1:a_size_neg
            revenue = 0
            if ap_ind != ind_a_zero
                for xp_ind in 1:x_size
                    pp_i, νp_i = x_grid[xp_ind,:]
                    earnings = pp_i # *variables_MIT.P[1,t+1]
                    if variables_MIT.V_good[ap_ind,xp_ind,3,t+1] >= variables_MIT.V_good[ap_ind,xp_ind,2,t+1]
                        revenue += Px[x_ind,xp_ind]*ξ*earnings
                    else
                        revenue += Px[x_ind,xp_ind]*(-a_grid_neg[ap_ind])
                    end
                end
                q_update = α*(revenue / ((1+r_b)*(-a_grid_neg[ap_ind]))) + (1-α)*variables_MIT.q[ap_ind,x_ind,1,t+1]
                variables_MIT.q[ap_ind,x_ind,1,t] = q_update < (1/(1+r_b)) ? q_update : 1/(1+r_b)
            else
                variables_MIT.q[ap_ind,x_ind,1,t] = 1/(1+r_f)
            end
        end
    end
    variables_MIT.q[:,:,2,t] = derivative_func(a_grid, variables_MIT.q[:,:,1,t])
    variables_MIT.q[:,:,3,t] = variables_MIT.q[:,:,1,t] .* repeat(a_grid, 1, x_size)
    variables_MIT.q[:,:,4,t] = derivative_func(a_grid, variables_MIT.q[:,:,3,t])
end

function LoM_func!(variables::mut_vars, parameters::NamedTuple)
    #---------------------------------------------------------------------#
    # compute the cross-sectional distribution and its transition matrix. #
    #---------------------------------------------------------------------#
    @unpack a_size, a_size_pos, ind_a_zero, x_size, Px, λ_H = parameters
    μ_size_good = x_size*a_size
    μ_size_bad = x_size*a_size_pos
    μ_size = μ_size_good + μ_size_bad
    variables.μ = spzeros(μ_size)
    variables.Pμ = spzeros(μ_size, μ_size)
    for x_ind in 1:x_size
        # good credit history
        ind_r1_good = 1 + (x_ind-1)*a_size
        ind_r2_good = x_ind*a_size
        variables.Pμ[ind_r1_good:ind_r2_good,1:x_size*a_size] = kron(transpose(Px[x_ind,:]),variables.policy_a_good_repay_matrix[ind_r1_good:ind_r2_good,:])
        variables.Pμ[ind_r1_good:ind_r2_good,(x_size*a_size+1):end] = kron(transpose(Px[x_ind,:]),variables.policy_a_good_default_matrix[ind_r1_good:ind_r2_good,:])
        # bad credit history
        ind_r1_bad = 1 + (x_ind-1)*a_size_pos
        ind_r2_bad = x_ind*a_size_pos
        ind_r1_bad_Pμ = ind_r1_bad + x_size*a_size
        ind_r2_bad_Pμ = ind_r2_bad + x_size*a_size
        for xp_ind in 1:x_size
            ind_c1_bad_Pμ = ind_a_zero + (xp_ind-1)*a_size
            ind_c2_bad_Pμ = xp_ind*a_size
            variables.Pμ[ind_r1_bad_Pμ:ind_r2_bad_Pμ,ind_c1_bad_Pμ:ind_c2_bad_Pμ] = λ_H*kron(transpose(Px[x_ind,xp_ind]),variables.policy_a_bad_matrix[ind_r1_bad:ind_r2_bad,:])
        end
        variables.Pμ[ind_r1_bad_Pμ:ind_r2_bad_Pμ,(x_size*a_size+1):end] = (1-λ_H)*kron(transpose(Px[x_ind,:]),variables.policy_a_bad_matrix[ind_r1_bad:ind_r2_bad,:])
    end
    MC = MarkovChain(variables.Pμ)
    SD = stationary_distributions(MC)
    variables.μ = SD[1]
end

function LoM_MIT_func!(t::Integer, variables_MIT::mut_vars_MIT, parameters::NamedTuple)
    #---------------------------------------------------------------------#
    # compute the cross-sectional distribution and its transition matrix. #
    #---------------------------------------------------------------------#
    @unpack a_size, a_size_pos, ind_a_zero, x_size, Px, λ_H = parameters
    μ_size_good = x_size*a_size
    μ_size_bad = x_size*a_size_pos
    μ_size = μ_size_good + μ_size_bad
    Pμ_temp = spzeros(μ_size, μ_size)
    policy_a_good_repay_matrix_temp = variables_MIT.policy_a_good_repay_matrix[t]
    policy_a_good_default_matrix_temp = variables_MIT.policy_a_good_default_matrix[t]
    policy_a_bad_matrix_temp = variables_MIT.policy_a_bad_matrix[t]
    for x_ind in 1:x_size
        # good credit history
        ind_r1_good = 1 + (x_ind-1)*a_size
        ind_r2_good = x_ind*a_size
        Pμ_temp[ind_r1_good:ind_r2_good,1:x_size*a_size] = kron(transpose(Px[x_ind,:]),policy_a_good_repay_matrix_temp[ind_r1_good:ind_r2_good,:])
        Pμ_temp[ind_r1_good:ind_r2_good,(x_size*a_size+1):end] = kron(transpose(Px[x_ind,:]),policy_a_good_default_matrix_temp[ind_r1_good:ind_r2_good,:])
        # bad credit history
        ind_r1_bad = 1 + (x_ind-1)*a_size_pos
        ind_r2_bad = x_ind*a_size_pos
        ind_r1_bad_Pμ = ind_r1_bad + x_size*a_size
        ind_r2_bad_Pμ = ind_r2_bad + x_size*a_size
        for xp_ind in 1:x_size
            ind_c1_bad_Pμ = ind_a_zero + (xp_ind-1)*a_size
            ind_c2_bad_Pμ = xp_ind*a_size
            Pμ_temp[ind_r1_bad_Pμ:ind_r2_bad_Pμ,ind_c1_bad_Pμ:ind_c2_bad_Pμ] = λ_H*kron(transpose(Px[x_ind,xp_ind]),policy_a_bad_matrix_temp[ind_r1_bad:ind_r2_bad,:])
        end
        Pμ_temp[ind_r1_bad_Pμ:ind_r2_bad_Pμ,(x_size*a_size+1):end] = (1-λ_H)*kron(transpose(Px[x_ind,:]),policy_a_bad_matrix_temp[ind_r1_bad:ind_r2_bad,:])
    end
    variables_MIT.Pμ[t] = Pμ_temp
    MC = MarkovChain(variables_MIT.Pμ[t])
    SD = stationary_distributions(MC)
    variables_MIT.μ[:,t] = SD[1]

end

function aggregate_func!(variables::mut_vars, parameters::NamedTuple)
    #------------------------------#
    # compute aggregate variables. #
    #------------------------------#
    @unpack L, a_grid, a_size, a_size_pos, ind_a_zero, x_size = parameters
    variables.A .= 0.0
    for x_ind in 1:x_size
        qap_good_itp = Spline1D(a_grid, variables.q[:,x_ind,1].*a_grid; k = 3, bc = "extrapolate")
        for a_ind in 1:a_size
            μ_ind_good = (x_ind-1)*a_size + a_ind
            ap_good = variables.policy_a_good[a_ind,x_ind,1]
            if ap_good < 0
                variables.A[1] += -qap_good_itp(ap_good)*variables.μ[μ_ind_good]
            else
                variables.A[2] += qap_good_itp(ap_good)*variables.μ[μ_ind_good]
            end
            if a_ind >= ind_a_zero
                a_ind_bad = a_ind - ind_a_zero + 1
                μ_ind_bad = (x_ind-1)*a_size_pos + a_ind_bad + x_size*a_size
                ap_bad = variables.policy_a_bad[a_ind_bad,x_ind]
                variables.A[2] += ap_bad*variables.μ[μ_ind_bad]
            end
        end
    end
    variables.A[3] = variables.A[1] - variables.A[2]
    variables.A[4] = variables.A[1] / variables.A[3]
end

function aggregate_MIT_func!(t::Integer, variables_MIT::mut_vars_MIT, parameters::NamedTuple)
    #------------------------------#
    # compute aggregate variables. #
    #------------------------------#
    @unpack L, a_grid, a_size, a_size_pos, ind_a_zero, x_size = parameters
    for x_ind in 1:x_size
        qap_good_itp = Spline1D(a_grid, variables_MIT.q[:,x_ind,1,t+1].*a_grid; k = 3, bc = "extrapolate")
        for a_ind in 1:a_size
            μ_ind_good = (x_ind-1)*a_size + a_ind
            ap_good = variables_MIT.policy_a_good[a_ind,x_ind,1,t]
            if ap_good < 0
                variables_MIT.A[1,t] += -qap_good_itp(ap_good)*variables_MIT.μ[μ_ind_good,t]
            else
                variables_MIT.A[2,t] += qap_good_itp(ap_good)*variables_MIT.μ[μ_ind_good,t]
            end
            if a_ind >= ind_a_zero
                a_ind_bad = a_ind - ind_a_zero + 1
                μ_ind_bad = (x_ind-1)*a_size_pos + a_ind_bad + x_size*a_size
                ap_bad = variables_MIT.policy_a_bad[a_ind_bad,x_ind,t]
                variables_MIT.A[2,t] += ap_bad*variables_MIT.μ[μ_ind_bad,t]
            end
        end
    end
    variables_MIT.A[3,t] = variables_MIT.A[1,t] - variables_MIT.A[2,t]
    variables_MIT.A[4,t] = variables_MIT.A[1,t] / variables_MIT.A[3,t]
end

function solve_func!(variables::mut_vars, parameters::NamedTuple; tol = 1E-12, iter_max = 10000)
    # solve the household's maximization problem to obtain the converged value functions via the modified EGM by Fella (2014, JEDC), given price schedules

    # unpack parameters
    @unpack a_grid, a_size, a_grid_neg, a_grid_pos, ind_a_zero, x_grid, x_size, β, Px, λ_H, σ, ξ, r_f = parameters

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
            variables.V_bad[:,x_ind], variables.policy_a_bad[:,x_ind] = sols_func(β_adj, Px_i, λ_H*V_good_p_pos .+ (1-λ_H)*V_bad_p, a_grid_pos, q_i_pos, σ, r_f, earnings)

            # (2) good credit history with repayment
            # println("x_ind = $x_ind, good")
            variables.V_good[:,x_ind,2], variables.policy_a_good[:,x_ind,2] = sols_func(β_adj, Px_i, V_good_p, a_grid, q_i, σ, r_f, earnings; check_rbl = 1)

            # (3) good credit history with defaulting
            V_hat_bad = V_hat_func(β_adj, Px_i, V_bad_p)
            variables.V_good[:,x_ind,3] .= u_func(earnings*(1-ξ), σ) + V_hat_bad[1]

            # (4) good credit history
            variables.V_good[:,x_ind,1] = max.(variables.V_good[:,x_ind,2], variables.V_good[:,x_ind,3])
            variables.policy_a_good[:,x_ind,1] = [variables.V_good[a_ind,x_ind,2] >= variables.V_good[a_ind,x_ind,3] ? variables.policy_a_good[a_ind,x_ind,2] : variables.policy_a_good[a_ind,x_ind,3] for a_ind in 1:a_size]
        end

        # update price, its derivative, and size of bond
        price_func!(variables, parameters)

        # check convergence
        crit = max(norm(variables.V_good[:,:,1]-V_good_p, Inf), norm(variables.V_bad-V_bad_p, Inf))

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end

    # update policy matrices
    variables.policy_a_good_repay_matrix, variables.policy_a_good_default_matrix = policy_matrix_func(variables.policy_a_good[:,:,1], parameters.a_grid; V_good = variables.V_good)
    variables.policy_a_bad_matrix = policy_matrix_func(variables.policy_a_bad, parameters.a_grid)

    # update the cross-sectional distribution
    LoM_func!(variables, parameters)

    # compute aggregate variables
    aggregate_func!(variables, parameters)
    # println("The lower state of preference shock is $(parameters.ν_grid[1])")
    # println("The excess return is $(parameters.r_bf)")
    # println("The risk-free rate is $(parameters.r_f)")
    # println("Targeted leverage ratio is $(parameters.L) and the implied leverage ratio is $(variables.A[4])")
    ED = variables.A[1] - (parameters.L/(parameters.L-1))*variables.A[2]
    # println("Excess demand is $ED")
    return ED
end

function solve_MIT_func!(variables_MIT::mut_vars_MIT, variables::mut_vars, parameters::NamedTuple; T::Integer = 150, tol = 1E-8, iter_max = 10000)
    # solve the household's maximization problem to obtain the converged value functions via the modified EGM by Fella (2014, JEDC), given price schedules

    # unpack parameters
    @unpack a_grid, a_size, a_grid_neg, a_grid_pos, ind_a_zero, x_grid, x_size, β, Px, λ_B, λ_H, θ, ω, σ, ξ, r_f, r_bf = parameters

    # initialize the iteration number and criterion
    iter = 0
    crit = Inf
    prog = ProgressThresh(tol, "Solving transitional dynamics: ")

    # initialize the next-period value functions
    N_p = similar(variables_MIT.A[3,:])

    while crit > tol && iter < iter_max

        # println("iter = $iter")
        # copy the current value functions to the pre-specified containers
        α = 0.8
        if iter > 0
            variables_MIT.A[3,:] =  α.*N_p .+ (1-α).*variables_MIT.A[3,:]
            copyto!(N_p, variables_MIT.A[3,:])
            variables_MIT.A[:,end] .= variables.A
            variables_MIT.P[2,:] = variables_MIT.A[3,:] ./ [variables.A[3,end]; variables_MIT.A[3,1:(end-1)]]
            for t in T:-1:1
                if t == T
                    variables_MIT.P[3,t] = θ*variables.A[4]
                    variables_MIT.P[4,t] = β*(1-λ_B+λ_B*variables_MIT.P[3,t])
                    variables_MIT.P[5,t] = (1-λ_B*(1+r_f)*(1-variables.A[4])) / ((λ_B+ω*(1-λ_B))*variables.A[4]) - 1
                    variables_MIT.P[6,t] = variables_MIT.P[4,t]*(variables_MIT.P[5,t]-r_f)/θ
                else
                    variables_MIT.P[3,t] = θ*variables_MIT.A[4,t+1]
                    variables_MIT.P[4,t] = β*(1-λ_B+λ_B*variables_MIT.P[3,t])
                    variables_MIT.A[4,t] = (variables_MIT.P[4,t]*(variables_MIT.P[2,t]+ω*(1-λ_B)*(1+r_f))) / (λ_B*θ + ω*(1-λ_B)*(θ+variables_MIT.P[4,t]*(1+r_f)))
                    variables_MIT.P[5,t] = (variables_MIT.P[2,t]-λ_B*(1+r_f)*(1-variables_MIT.A[4,t])) / ((λ_B+ω*(1-λ_B))*variables_MIT.A[4,t]) - 1
                    variables_MIT.P[6,t] = variables_MIT.P[4,t+1]*(variables_MIT.P[5,t]-r_f)/θ
                end
            end
        else
            copyto!(N_p,variables_MIT.A[3,:])
        end

        for t in (T-1):-1:1
            # println("t = $t")
            V_bad_p = variables_MIT.V_bad[:,:,t+1]
            V_good_p = variables_MIT.V_good[:,:,1,t+1]
            V_good_p_pos = V_good_p[ind_a_zero:end,:]

            # start looping over each household's type
            for x_ind in 1:x_size
                # println("x = $x_ind")
                # abstract necessary variables
                Px_i = Px[x_ind,:]
                p_i, ν_i = x_grid[x_ind,:]
                q_i = variables_MIT.q[:,x_ind,:,t+1]

                # define two handy variables
                earnings = variables_MIT.P[1,t]*p_i
                β_adj = ν_i*β

                #-------------------------------------------------------------#
                # compute global solutions, update value and policy functions #
                #-------------------------------------------------------------#
                # (1) bad credit history
                # println("x_ind = $x_ind, bad")
                q_i_pos = q_i[ind_a_zero:end,:]
                variables_MIT.V_bad[:,x_ind,t], variables_MIT.policy_a_bad[:,x_ind,t] = sols_func(β_adj, Px_i, λ_H*V_good_p_pos .+ (1-λ_H)*V_bad_p, a_grid_pos, q_i_pos, σ, r_f, earnings)

                # (2) good credit history with repayment
                # println("x_ind = $x_ind, good")
                variables_MIT.V_good[:,x_ind,2,t], variables_MIT.policy_a_good[:,x_ind,2,t] = sols_func(β_adj, Px_i, V_good_p, a_grid, q_i, σ, r_f, earnings; check_rbl = 1)

                # (3) good credit history with defaulting
                V_hat_bad = V_hat_func(β_adj, Px_i, V_bad_p)
                variables_MIT.V_good[:,x_ind,3,t] .= u_func(earnings*(1-ξ), σ) + V_hat_bad[1]

                # (4) good credit history
                variables_MIT.V_good[:,x_ind,1,t] = max.(variables_MIT.V_good[:,x_ind,2,t], variables_MIT.V_good[:,x_ind,3,t])
                variables_MIT.policy_a_good[:,x_ind,1,t] = [variables_MIT.V_good[a_ind,x_ind,2,t] >= variables_MIT.V_good[a_ind,x_ind,3,t] ? variables_MIT.policy_a_good[a_ind,x_ind,2,t] : variables_MIT.policy_a_good[a_ind,x_ind,3,t] for a_ind in 1:a_size]
            end

            # update price, its derivative, and size of bond
            price_MIT_func!(t, variables_MIT, parameters)

            # update policy matrices
            variables_MIT.policy_a_good_repay_matrix[t], variables_MIT.policy_a_good_default_matrix[t] = policy_matrix_func(variables_MIT.policy_a_good[:,:,1,t], parameters.a_grid; V_good = variables_MIT.V_good[:,:,:,t])
            variables_MIT.policy_a_bad_matrix[t] = policy_matrix_func(variables_MIT.policy_a_bad[:,:,t], parameters.a_grid)

            # update the cross-sectional distribution
            LoM_MIT_func!(t, variables_MIT, parameters)

            # compute aggregate variables
            aggregate_MIT_func!(t, variables_MIT, parameters)
        end

        # check convergence
        crit = norm(variables_MIT.A[3,:] .- N_p, Inf)

        # report preogress
        ProgressMeter.update!(prog, crit)

        # update the iteration number
        iter += 1
    end
end
