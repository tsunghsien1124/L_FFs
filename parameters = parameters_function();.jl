using BenchmarkTools

parameters = initialize_parameters();
variables = create_variables(parameters);
itp_cache = build_itp_cache(variables, parameters);
variables.EV[:,1,1,1] .= rand(parameters.a_size)
@btime itp_cache.EV[1,1,1].itp.coefs .= variables.EV[:,1,1,1]

# V_p = rand(Float64, size(similar(variables.V)));
# V_pos_p = rand(Float64, size(similar(variables.V_pos)));
# @btime E_V_function!($V_p, $V_pos_p, $variables, $parameters);

e1_i = parameters.e1_size - 1
e2_i = parameters.e2_size - 1
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, e2_i, e1_i], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, e2_i, e1_i] .* parameters.a_grid_neg, seriestype=:scatter)
plot!([variables.rbl_a[e2_i, e1_i]], [variables.rbl_qa[e2_i, e1_i]], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, e1_i])

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, e1_i] .* parameters.a_grid_neg)
plot!(variables.rbl_a[:, e1_i], variables.rbl_qa[:, e1_i], seriestype=:scatter)


@views p = variables.q[1:(parameters.a_ind_zero-1), e2_i, e1_i]
@views a = parameters.a_grid_neg[1:(end-1)]
a_test_1 = collect(parameters.a_min:1:0.00001)
a_test_2 = collect(parameters.a_min:0.001:0.00001)
σ(x) = 1 / (1 + exp(-x))
logit(p) = log(p) - log1p(-p)
S = logit.(clamp.(p, 1E-8, 1-1E-8))
Sitp = PCHIPInterpolation(S, a)
Sitp_linear = LinearInterpolation(p, a)

plot(a, p)
plot!(a_test_1, Sitp_linear.(a_test_1), seriestype=:scatter)

plot(a, p)
plot!(a_test_2, Sitp_linear.(a_test_2))

plot(a, p)
plot!(a_test_1, σ.(Sitp.(a_test_1)), seriestype=:scatter)

plot(a, p)
plot!(a_test_2, σ.(Sitp.(a_test_2)))

using DataInterpolations, BenchmarkTools

σ(x) = inv(1 + exp(-x))
logit(p) = log(p) - log1p(-p)

x = parameters.a_grid_neg[1:(end-1)]
y = p
y_logit = logit.(clamp.(y, 1e-12, 1-1e-12))

lin = LinearInterpolation(y, x)
pch = PCHIPInterpolation(y, x)
pch_logit = PCHIPInterpolation(y_logit, x)

t = -0.123  # some query inside [x[1], x[end]]

@btime $lin($t)
@btime $pch($t)
@btime σ($pch_logit($t))

############
using Interpolations
A = reshape(collect(1.0:5.0), 5, 1, 1, 1)
itp = linear_interpolation(collect(1:5), @view(A[:,1,1,1]), extrapolation_bc=Line())
v1 = itp(3.0)            # ~3.0
A[:,1,1,1] .= 100:104
v2 = itp(3.0)            # still ~3.0  ← internally copied at construction
itp.itp.coefs .= @view A[:,1,1,1]
v3 = itp(3.0)            # now ~102.0  ← manual refresh fixed it