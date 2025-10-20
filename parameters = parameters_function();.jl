#=================#
# Import packages #
#=================#
using Distributions, StatsFuns, QuadGK
# using JLD2: @save, @load
using LinearAlgebra
using Optim
using Parameters: @unpack
using PrettyTables
using ProgressMeter
using QuantEcon: rouwenhorst, tauchen, stationary_distributions, MarkovChain
# using Roots
# using CSV
# using Tables
using Plots
using Random123
# using GLM
# using DataFrames
# using Measures
using BenchmarkTools, Profile
using Polyester
using Interpolations
# using FastGaussQuadrature
# using LoopVectorization

static_parameters = initialize_static_parameters();
tuned_parameters = initialize_tuned_parameters(static_parameters; λ = 0.0, Ph=1.0 / 6.0, κ=697.0 / 33176.0, β=0.9540, η=0.2400, ψ=0.970, θ=1.0 / 3.50, ζ=0.0167); # kwargs = (β = 0.99, λ = 0.01) ; kwargs...
parameters = (; static_parameters..., tuned_parameters...);
variables = create_variables(parameters);
itp_cache = build_itp_cache(variables, parameters);
simul_itp_cache = build_simul_itp_cache(variables, parameters);
simul_panel = initialize_panel(num_households=80_000, num_periods=2_500);
solve_economy_function!(variables, itp_cache, simul_panel, simul_itp_cache, parameters);

crit_VP_old, parameters_old, variables_old, simul_panel_old, flag_old = optimal_multiplier_function(Ph=1.0 / 6.0, κ=697.0 / 33176.0, β=0.9540, η=0.2200, ψ=0.970, θ=1.0 / 3.85, ζ=0.0160);
# crit_VP_new, parameters_new, variables_new, simul_panel_new, flag_new = optimal_multiplier_function(Ph=1.0 / 10.0, κ=975.0 / 33176.0, β=0.9540, η=0.2200, ψ=0.970, θ=1.0 / 3.50, ζ=0.0160);

# variables = create_variables(parameters_new);
# itp_cache = build_itp_cache(variables, parameters_new);
# crit_VP = solve_value_and_policy_functions!(variables, itp_cache, parameters_new; tol=1E-6, relax_V=1.0, relax_q=1.0, bellman_step=1)


# solve_value_and_policy_functions!(variables, itp_cache, parameters; tol=1E-6, relax=1.0, bellman_step=1);
# update_simul_itp_cache!(simul_itp_cache, variables, parameters);
# simulate_household_panel!(simul_panel, parameters, simul_itp_cache);
# compute_moments!(variables, parameters, simul_panel; burnin=500);

a_range = range(-3, 20, length=101)
# histogram(reshape(simul_panel.asset_state[1001:end, :], :, 1), bins=a_range, normalize=:pdf, color=:blue)
histogram(reshape(simul_panel.asset_state[end, :], :, 1), bins=a_range, normalize=:pdf, color=:blue)

# V_p = rand(Float64, size(similar(variables.V)));
# V_pos_p = rand(Float64, size(similar(variables.V_pos)));
# @btime E_V_function!($V_p, $V_pos_p, $variables, $parameters);

plot(1:parameters.a_size_neg, parameters.a_grid_neg, seriestype=:scatter)
plot(1:parameters.a_size_pos, parameters.a_grid_pos, seriestype=:scatter)

e1_i = parameters.e1_size
e2_i = parameters.e2_size
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, e2_i, e1_i], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, e2_i, e1_i] .* parameters.a_grid_neg, seriestype=:scatter)
plot!([variables.rbl_a[e2_i, e1_i]], [variables.rbl_qa[e2_i, e1_i]], seriestype=:scatter)

plot(parameters.a_grid_neg[25:end], variables.q[25:parameters.a_size_neg, :, 1])
plot(parameters.a_grid_neg[25:end], variables.q[25:parameters.a_size_neg, :, 1], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, 1])
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, 1], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, end-1])
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, end-1], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, end])
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, end], seriestype=:scatter)


plot(parameters_old.a_grid_neg, variables_old.q[1:parameters_old.a_size_neg, :, 1])
plot(parameters_old.a_grid_neg, variables_old.q[1:parameters_old.a_size_neg, :, end-1])
plot(parameters_old.a_grid_neg, variables_old.q[1:parameters_old.a_size_neg, :, end])
plot(parameters_old.a_grid_neg, variables_old.q[1:parameters_old.a_size_neg, :, end] .* parameters_old.a_grid_neg)

plot(parameters_old.a_grid_neg[25:end], variables_old.q[25:parameters_old.a_size_neg, :, 1])
plot!(parameters_new.a_grid_neg[25:end], variables_new.q[25:parameters_new.a_size_neg, :, 1], ls=:dash)

plot(parameters_old.a_grid_neg, variables_old.q[1:parameters_old.a_size_neg, :, end])
plot!(parameters_new.a_grid_neg, variables_new.q[1:parameters_new.a_size_neg, :, end], ls=:dash)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, 1] .* parameters.a_grid_neg)
plot!(variables.rbl_a[:, 1], variables.rbl_qa[:, 1], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, end-1] .* parameters.a_grid_neg)
plot!(variables.rbl_a[:, end-1], variables.rbl_qa[:, end-1], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, end] .* parameters.a_grid_neg)
plot!(variables.rbl_a[:, end], variables.rbl_qa[:, end], seriestype=:scatter)


plot(parameters.a_grid[parameters.a_ind_zero:parameters.a_ind_zero+185], variables.V[parameters.a_ind_zero:parameters.a_ind_zero+185,2,1,:])
plot(parameters.a_grid[parameters.a_ind_zero:parameters.a_ind_zero+180], variables.V[parameters.a_ind_zero:parameters.a_ind_zero+180,2,1,:], seriestype=:scatter)

#####
e1_i = parameters.e1_size
e2_i = parameters.e2_size
ν_i = 1 # parameters.ν_size
e3_i = 1:2 # parameters.e3_size

plot(parameters.a_grid_neg, variables.V_nd[1:parameters.a_size_neg, e3_i, ν_i, e2_i, e1_i])
hline!([variables.V_d[e3_i, ν_i, e2_i, e1_i]])
scatter!([variables.thres_a[e3_i, ν_i, e2_i, e1_i]], [variables.V_d[e3_i, ν_i, e2_i, e1_i]])

plot(parameters.a_grid_neg, inverse_utility.(variables.V_nd[1:parameters.a_size_neg, e3_i, ν_i, e2_i, e1_i], parameters.γ))
hline!([inverse_utility.(variables.V_d[e3_i, ν_i, e2_i, e1_i], parameters.γ)])
scatter!([variables.thres_a[e3_i, ν_i, e2_i, e1_i]], [inverse_utility.(variables.V_d[e3_i, ν_i, e2_i, e1_i], parameters.γ)])

a_p_i = 50 # parameters.a_size_neg
a_p_ = parameters.a_grid_neg[a_p_i]
e1_ = parameters.e1_grid[e1_i]
e3_ = parameters.e3_grid[e3_i]

plot(parameters.e2_grid, variables.thres_a[e3_i, ν_i, :, e1_i])
scatter!([variables.thres_e2[a_p_i, e3_i, ν_i, e1_i]], [parameters.a_grid_neg[a_p_i]])

plot(exp.(parameters.e2_grid), variables.thres_a[e3_i, ν_i, :, e1_i])
scatter!([exp(variables.thres_e2[a_p_i, e3_i, ν_i, e1_i])], [parameters.a_grid_neg[a_p_i]])

W_ = parameters.w_λ * exp(variables.thres_e2[a_p_i, e3_i, ν_i, e1_i] + e1_ + e3_)
plot(parameters.W[e3_i, :, e1_i], variables.thres_a[e3_i, ν_i, :, e1_i])
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

plot(parameters.a_grid_neg[80:end], variables.policy_d[80:parameters.a_size_neg, e3_i, ν_i, e2_i, :], seriestype=:scatter)

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
S = logit.(clamp.(p, 1E-8, 1 - 1E-8))
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
y_logit = logit.(clamp.(y, 1e-12, 1 - 1e-12))

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
itp = linear_interpolation(collect(1:5), @view(A[:, 1, 1, 1]), extrapolation_bc=Line())
v1 = itp(3.0)            # ~3.0
A[:, 1, 1, 1] .= 100:104
v2 = itp(3.0)            # still ~3.0  ← internally copied at construction
itp.itp.coefs .= @view A[:, 1, 1, 1]
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
Γ_sample = rand(3, 4, 5, 6)
V_sample = rand(3, 4, 5, 6)

# Single-threaded comparison
@btime dot($Γ_sample, $V_sample)
@btime sum($Γ_sample .* $V_sample)