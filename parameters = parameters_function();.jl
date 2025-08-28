#=================#
# Import packages #
#=================#
# using Dierckx
# using FLOWMath
using Distributions
using QuadGK
using JLD2: @save, @load
using LinearAlgebra
BLAS.set_num_threads(1)
using Optim
# using BlackBoxOptim
using Parameters: @unpack
using PrettyTables
using ProgressMeter
using QuantEcon: gridmake, rouwenhorst, tauchen, stationary_distributions, MarkovChain
using Roots
using CSV
using Tables
using Plots
# using Random
using Random123
using GLM
using DataFrames
using Measures
using BenchmarkTools, Profile
using Polyester
using Interpolations
using FastGaussQuadrature
using LoopVectorization
# using DataInterpolations
using StatsFuns

parameters = initialize_parameters(a_size_neg = 201, a_degree_neg = 2, a_degree_pos = 2, λ = 0.00);
variables = create_variables(parameters);
itp_cache = build_itp_cache(variables, parameters);
solve_value_and_pricing_function!(variables, parameters, itp_cache; slow_updating = 1.0);

simul_itp_cache = build_simul_itp_cache(variables, parameters);
simul_panel = initialize_panel(num_households = 50000, num_periods = 2000);
simulate_household_panel!(parameters, simul_itp_cache, simul_panel);

histogram(reshape(simul_panel.asset_state[1001:end,:],:,1))

# V_p = rand(Float64, size(similar(variables.V)));
# V_pos_p = rand(Float64, size(similar(variables.V_pos)));
# @btime E_V_function!($V_p, $V_pos_p, $variables, $parameters);

plot(1:parameters.a_size_neg, parameters.a_grid_neg, seriestype=:scatter)

e1_i = parameters.e1_size
e2_i = parameters.e2_size
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, e2_i, e1_i], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, e2_i, e1_i] .* parameters.a_grid_neg, seriestype=:scatter)
plot!([variables.rbl_a[e2_i, e1_i]], [variables.rbl_qa[e2_i, e1_i]], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, 1])

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, end])

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, 1] .* parameters.a_grid_neg)
plot!(variables.rbl_a[:, 1], variables.rbl_qa[:, 1], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, end] .* parameters.a_grid_neg)
plot!(variables.rbl_a[:, end], variables.rbl_qa[:, end], seriestype=:scatter)

#####
e1_i = parameters.e1_size
e2_i = 2 #parameters.e2_size
ν_i = parameters.ν_size
e3_i = parameters.e3_size

plot(parameters.a_grid_neg, variables.V_nd[1:parameters.a_size_neg, e3_i, ν_i, e2_i, e1_i])
hline!([variables.V_d[e3_i, ν_i, e2_i, e1_i]])
scatter!([variables.thres_a[e3_i, ν_i, e2_i, e1_i]], [variables.V_d[e3_i, ν_i, e2_i, e1_i]])

plot(parameters.a_grid_neg, inverse_utility.(variables.V_nd[1:parameters.a_size_neg, e3_i, ν_i, e2_i, e1_i], parameters.γ))
hline!([inverse_utility(variables.V_d[e3_i, ν_i, e2_i, e1_i], parameters.γ)])
scatter!([variables.thres_a[e3_i, ν_i, e2_i, e1_i]], [inverse_utility(variables.V_d[e3_i, ν_i, e2_i, e1_i], parameters.γ)])

a_p_i = 50 # parameters.a_size_neg
a_p_ = parameters.a_grid_neg[a_p_i]
e1_ = parameters.e1_grid[e1_i]
e3_ = parameters.e3_grid[e3_i]

plot(parameters.e2_grid, variables.thres_a[e3_i, ν_i, :, e1_i])
scatter!([variables.thres_e2[a_p_i, e3_i, ν_i, e1_i]], [parameters.a_grid_neg[a_p_i]])

plot(exp.(parameters.e2_grid), variables.thres_a[e3_i, ν_i, :, e1_i])
scatter!([exp(variables.thres_e2[a_p_i, e3_i, ν_i, e1_i])], [parameters.a_grid_neg[a_p_i]])

W_ = parameters.w_λ * exp(variables.thres_e2[a_p_i, e3_i, ν_i, e1_i] + e1_ + e3_)
plot(parameters.W[e3_i,:,e1_i], variables.thres_a[e3_i, ν_i, :, e1_i])
scatter!([W_], [parameters.a_grid_neg[a_p_i]])

plot(parameters.a_grid_neg, exp.(variables.thres_e2[:, e3_i, ν_i, :]))

#####

e1_i = 1 # parameters.e1_size
e2_i = 1 # parameters.e2_size
ν_i = 1 # parameters.ν_size
e3_i = 1 # parameters.e3_size

plot(parameters.a_grid_pos, variables.V[parameters.a_ind_zero:end, e3_i, ν_i, :, e1_i])

# plot(parameters.a_grid_pos, V_p[parameters.a_ind_zero:end, e3_i, ν_i, :, e1_i])

plot(parameters.a_grid_pos[1:10], variables.V[parameters.a_ind_zero+1:parameters.a_ind_zero+10, e3_i, ν_i, :, e1_i])

plot(parameters.a_grid_neg[1:30], variables.V[1:30, e3_i, ν_i, :, e1_i])

plot(parameters.a_grid, variables.EV[:, ν_i, e2_i, e1_i])

plot(parameters.a_grid_neg, variables.EV[1:parameters.a_size_neg, ν_i, e2_i, e1_i])

plot(parameters.a_grid_neg[1:30], variables.EV[1:30, ν_i, e2_i, e1_i])

plot(parameters.a_grid_pos, variables.EV_pos[:, ν_i, e2_i, e1_i])

plot(parameters.a_grid_pos, variables.EV_Ph[:, ν_i, e2_i, e1_i])

#####

e1_i = 1 # parameters.e1_size
e2_i = 1 # parameters.e2_size
ν_i = 1 # parameters.ν_size
e3_i = 1 # parameters.e3_size

plot(parameters.a_grid_neg, variables.policy_a[1:parameters.a_size_neg, e3_i, ν_i, e2_i, :])

plot(parameters.a_grid_neg, variables.policy_a[1:parameters.a_size_neg, e3_i, ν_i, :, e1_i])

plot(parameters.a_grid_neg, variables.policy_a[1:parameters.a_size_neg, e3_i, :, e2_i, e1_i])

plot(parameters.a_grid_neg, variables.policy_a[1:parameters.a_size_neg, :, ν_i, e2_i, e1_i])

plot(parameters.a_grid_neg[60:end], variables.policy_a[60:parameters.a_size_neg, :, ν_i, e2_i, e1_i])

plot(parameters.a_grid_neg[60:end], variables.policy_a[60:parameters.a_size_neg, e3_i, :, e2_i, e1_i])

plot(parameters.a_grid_neg, variables.policy_a[1:parameters.a_size_neg, e3_i, ν_i, e2_i, :])

plot(parameters.a_grid_neg, variables.policy_d[1:parameters.a_size_neg, e3_i, ν_i, e2_i, :])

plot(parameters.a_grid_neg, variables.policy_d[1:parameters.a_size_neg, e3_i, :, e2_i, e1_i])

plot(parameters.a_grid, variables.policy_a[:, e3_i, ν_i, :, e1_i])

plot(parameters.a_grid_pos[1:20], variables.policy_a_pos[1:20, e3_i, ν_i, :, e1_i])

#####

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

#####

# Before the function, check for duplicates
function check_index_uniqueness(loop_indices, name)
    indices_set = Set()
    for idx in loop_indices
        key = idx.I  # or whatever indexing you're using
        if key in indices_set
            println("Duplicate found in $name: $key")
            return false
        end
        push!(indices_set, key)
    end
    return true
end

# Call before your loops
check_index_uniqueness(parameters.loop_a_ν_e2_e1, "loop_a_ν_e2_e1")
check_index_uniqueness(parameters.loop_a_pos_ν_e2_e1, "loop_a_pos_ν_e2_e1")

#####

using BenchmarkTools

# Test with your typical array sizes
Γ_sample = rand(3,4,5,6)
V_sample = rand(3,4,5,6)

# Single-threaded comparison
@btime dot($Γ_sample, $V_sample)
@btime sum($Γ_sample .* $V_sample)