
function rbl(x_i::Integer, q::Array{Float64,2}, parameters::NamedTuple)
    # compute the risky borrowing limit for type (x_i)
    @unpack a_grid_neg, a_size_neg = parameters
    qa_func = LinearInterpolation(a_grid_neg, q(1:a_size_neg,x_i).*a_grid_neg)
    results = optimize(qa_func, a_size_neg[1], 0)
    rbl = results.minimizer
    return rbl
end

function V_hat(ap_i::Integer, x_i::Integer, V::Array{Float64,2}, parameters::NamedTuple)
    # compute the discounted expected value function for asset holding in the next period (ap_i) and type (x_i)
    @unpack β, Px = parameters
    V_hat = β*Px[x_i,:]*V[ap_i,:]
    return V_hat
end

function DV_hat(ap_i::Integer, x_i::Integer, V::Array{Float64,2}, parameters::NamedTuple)
    # compute first-order derivative of the discounted expected value function for asset holding in the next period (ap_i) and type (x_i) wrt the first argument (ap_i) by forward finite difference
    if ap_i < size(V,1)
        DV_hat = (V_hat(ap_i+1,x_i,V,parameters) - V_hat(ap_i,x_i,V,parameters)) / (a_grid[ap_i+1] - a_grid[ap_i])
    else
        DV_hat = (V_hat(ap_i,x_i,V,parameters) - V_hat(ap_i-1,x_i,V,parameters)) / (a_grid[ap_i] - a_grid[ap_i-1])
    return DV_hat
end
