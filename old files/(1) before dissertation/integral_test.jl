using QuadGK

f1(x) = exp(-x^2)
integral1, err1 = quadgk(x -> f1(x), 0, 1, rtol=1E-8)

f2(x) = (1/sqrt(2*π))*exp(-x^2/2.0)
integral2, err2 = quadgk(x -> f2(x), -Inf, Inf, rtol=1E-8)

f3(x) = x*(1/sqrt(2*π))*exp(-x^2/2.0)
integral3, err3 = quadgk(x -> f3(x), -Inf, Inf, order=100, rtol=1E-10)

a_p_i = 1
V_diff = variables.V_nd[a_p_i,:,1] .- variables.V_d[:,1] .- ξ_bar
if all(V_diff .> 0.0)
    e_p_thres = -Inf
elseif all(V_diff .< 0.0)
    e_p_thres = Inf
else
    e_p_lower = e_grid[findall(V_diff .<= 0.0)[end]]
    e_p_upper = e_grid[findall(V_diff .>= 0.0)[1]]
    # V_diff_itp = Akima(e_grid, V_diff)
    V_nd_itp = Akima(e_grid, variables.V_nd[a_p_i,:,1])
    V_d_itp = Akima(e_grid, variables.V_d[:,1])
    V_diff_itp(x) = V_nd_itp(x) - V_d_itp(x) - ξ_bar
    e_p_thres = find_zero(e_p->V_diff_itp(e_p), (e_p_lower, e_p_upper), Bisection())
end

e_i = 1
e = e_grid[e_i]
d =  Akima(e_grid, variables.policy_d[a_p_i,:,1])
dist = Normal(e_ρ*e, e_σ)
f4(x) = (1-d(x))*pdf(dist, x)
integral4, err4 = quadgk(x -> f4(x), -Inf, Inf, order=100, rtol=1E-10)

using Main.FLOWMath: Akima, akima, interp2d
q_itp(ap, e) = interp2d(akima, parameters.a_grid_neg, parameters.e_grid, variables.q, [ap], [e])[]

variables.q
