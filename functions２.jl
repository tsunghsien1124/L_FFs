function u(c::Real, σ::Real)
    # compute utility
    if c > 0
        if σ == 1
            return log(c)
        else
            return 1 / ((1-σ)*c^(σ-1))
        end
    else
        println("non-positive consumption!")
    end
end

function du(c::Real, σ::Real)
    # compute marginal utility
    return c^(-σ)
end

function inv_du(x::Real, σ::Real)
    # compute marginal utility
    return x^(-1/σ)
end

function rbl(x_i::Integer, q::Array{Float64,2}, parameters::NamedTuple)
    # compute the risky borrowing limit for curren type (x_i)
    @unpack a_grid_neg, a_size_neg = parameters
    qa_func = LinearInterpolation(a_grid_neg, q(1:a_size_neg,x_i).*a_grid_neg)
    results = optimize(qa_func, a_grid_neg[1], 0)
    rbl = results.minimizer
    return rbl
end

function V_hat(ap_i::Integer, x_i::Integer, V::Array{Float64,2}, parameters::NamedTuple)
    # compute the discounted expected value function for asset holding in the next period (ap_i) and current type (x_i)
    @unpack β, Px = parameters
    V_hat = β*Px[x_i,:]*V[ap_i,:]
    return V_hat
end

function dV_hat(ap_i::Integer, x_i::Integer, V::Array{Float64,2}, a_grid::Array{Float64,1}, parameters::NamedTuple)
    # compute first-order derivative of the discounted expected value function for asset holding in the next period (ap_i) and current type (x_i) wrt the first argument (ap_i) by forward finite difference
    if ap_i < size(V,1)
        dV_hat = (V_hat(ap_i+1,x_i,V,parameters) - V_hat(ap_i,x_i,V,parameters)) / (a_grid[ap_i+1] - a_grid[ap_i])
    else
        dV_hat = (V_hat(ap_i,x_i,V,parameters) - V_hat(ap_i-1,x_i,V,parameters)) / (a_grid[ap_i] - a_grid[ap_i-1])
    return dV_hat
end

function ncr(x_i::Integer, V::Array{Float64,2}, a_grid::Array{Float64,1}, parameters::NamedTuple)
    # compute the non-concave region for type (x_i)
    a_size = length(a_grid)
    ncr = [1 a_size]
    dV_hat_vec = dV_hat.(1:a_size, x_i, V, a_grid, parameters)
    # (1) find the lower bound
    V_max, i_max = dV_hat_vec[1], 1
    while i_max < a_size
        if V_max > maximum(dV_hat_vec[(i_max+1):end])
            i_max += 1
            V_max = dV_hat_vec[i_max]
        else
            ncr[1] = i_max
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
            ncr[2] = i_min
            break
        end
    end
    return ncr
end

function CoH_G(ap_i::Integer, x_i::Integer, V::Array{Float64,2}, a_grid::Array{Float64,1}, q::Array{Float64,2}, parameters::NamedTuple)
    # compute the cash on hands for the case of asset holding in the next period (ap_i), current type (x_i), and good credit history with repayment. Note that cash on hands for the case of good credit history with defaulting is trivially determined so it can be ignored
    return inv_du(dV_hat(ap_i, x_i, V, a_grid, parameters)/q[ap_i,x_i]) + q[ap_i,x_i]*a_grid[ap_i]
end

function CoH_B(ap_i::Integer, x_i::Integer, V::Array{Float64,2}, a_grid_pos::Array{Float64,1}, q::Array{Float64,2}, parameters::NamedTuple)
    # compute the cash on hands for asset holding in the next period (ap_i), current type (x_i), and bad credit hostory. Note that ap must be positive (saving only)
    @unpack λ = parameters
    return inv_du(λ*dV_hat(ap_i, x_i, V_good, a_grid_pos, parameters)+(1-λ)*dV_hat(ap_i, x_i, V_bad, a_grid_pos, parameters)) + a_grid_pos[ap_i]
end
