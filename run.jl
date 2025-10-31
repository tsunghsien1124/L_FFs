#===============================#
# Import packages and functions #
#===============================#
using Distributions, StatsFuns, QuadGK
using LinearAlgebra
using Optim
using Parameters: @unpack
using PrettyTables
using ProgressMeter
using QuantEcon: rouwenhorst, tauchen, stationary_distributions, MarkovChain
using Random123
using BenchmarkTools, Profile
using Polyester
using Interpolations
include("solving_stationary_equilibrium.jl")

#===========#
# Benchmark #
#===========#
# crit_VP_old, parameters_old, variables_old, simul_panel_old, flag_old = optimal_multiplier_function(Ph=1.0 / 6.0, κ=715.0 / 33176.0, β=0.9550, η=0.2250, ψ=0.970, θ=1.0 / 3.95, ζ=0.0162);
λ_opt_old = 0.003805307419379075
crit_VP_old, parameters_old, variables_old, simul_panel_old, flag_old = optimal_multiplier_function(λ_opt=λ_opt_old, Ph=1.0 / 6.0, κ=715.0 / 33176.0, β=0.9550, η=0.2250, ψ=0.970, θ=1.0 / 3.95, ζ=0.0162);
mnts_ag_old, mnts_e1_old, mnts_e2_old, mnts_e3_old = compute_group_moments(simul_panel_old, parameters_old; burnin=500);
# mnts_ag_low_e1_old, mnts_ag_mid_e1_old, mnts_ag_hig_e1_old, mnts_e1_low_e2_old, mnts_e1_mid_e2_old, mnts_e1_hig_e2_old

a_grid_neg_μ, a_size_neg_μ, a_grid_pos_μ, a_size_pos_μ, a_grid_μ, a_size_μ, qa_grid_μ = density_agrid(variables_old, simul_panel_old, parameters_old; a_size_neg_μ = 5001, a_size_pos_μ = 5001);
μ_old_1 = panel_to_density(simul_panel_old, a_grid_μ; burnin=500);
qμ_old_1 = μ_old_1 .* reshape(qa_grid_μ, (a_size_μ, 1, parameters_old.e2_size, parameters_old.e1_size, 1));
variables_old.aggregate_variables.L + sum(qμ_old_1[1:a_size_neg_μ, :, :, :, :])
variables_old.aggregate_variables.D - sum(qμ_old_1[(a_size_neg_μ+1):end, :, :, :, :])

# μ_old_1 = panel_to_density(simul_panel_old, parameters_old.a_grid; burnin=500);
# qμ_old_1 = μ_old_1 .* reshape(variables_old.q, (parameters_old.a_size, 1, parameters_old.e2_size, parameters_old.e1_size, 1));
# variables_old.aggregate_variables.L + sum(qμ_old_1[1:parameters_old.a_size_neg, :, :, :, :] .* reshape(parameters_old.a_grid_neg, (parameters_old.a_size_neg, 1, 1, 1, 1)))
# variables_old.aggregate_variables.D - sum(qμ_old_1[parameters_old.a_ind_zero:end, :, :, :, :] .* reshape(parameters_old.a_grid_pos, (parameters_old.a_size_pos, 1, 1, 1, 1)))

#========#
# BAPCPA #
#========#
# crit_VP_new, parameters_new, variables_new, simul_panel_new, flag_new = optimal_multiplier_function(Ph=1.0 / 10.0, κ=991.0 / 33176.0, β=0.9550, η=0.2250, ψ=0.970, θ=1.0 / 3.95, ζ=0.0162);
λ_opt_new = 0.009691772060120379
crit_VP_new, parameters_new, variables_new, simul_panel_new, flag_new = optimal_multiplier_function(λ_opt=λ_opt_new, Ph=1.0 / 10.0, κ=991.0 / 33176.0, β=0.9550, η=0.2250, ψ=0.970, θ=1.0 / 3.95, ζ=0.0162);
mnts_ag_new, mnts_e1_new, mnts_e2_new, mnts_e3_new = compute_group_moments(simul_panel_new, parameters_new; burnin=500);

#=================#
# BAPCPA (No FFs) #
#=================#
ι_λ_, w_λ_ = parameters_old.ι_λ, parameters_old.w_λ
crit_VP_new_NFFs, parameters_new_NFFs, variables_new_NFFs, simul_panel_new_NFFs, flag_new_NFFs = optimal_multiplier_function(Ph=1.0 / 10.0, κ=991.0 / 33176.0, β=0.9550, η=0.2250, ψ=0.970, θ=1.0 / Inf, ζ=0.0162, ι_λ_=ι_λ_, w_λ_=w_λ_);
mnts_ag_new_NFFs, mnts_e1_new_NFFs, mnts_e2_new_NFFs, mnts_e3_new_NFFs = compute_group_moments(simul_panel_new_NFFs, parameters_new_NFFs; burnin=500);

#=========================#
# BAPCPA (No FFs) -- wage #
#=========================#
ι_λ_, w_λ_ = parameters_old.ι_λ, parameters_new.w_λ
crit_VP_new_NFFs_w, parameters_new_NFFs_w, variables_new_NFFs_w, simul_panel_new_NFFs_w, flag_new_NFFs_w = optimal_multiplier_function(Ph=1.0 / 10.0, κ=991.0 / 33176.0, β=0.9550, η=0.2250, ψ=0.970, θ=1.0 / Inf, ζ=0.0162, ι_λ_=ι_λ_, w_λ_=w_λ_);
mnts_ag_new_NFFs_w, mnts_e1_new_NFFs_w, mnts_e2_new_NFFs_w, mnts_e3_new_NFFs_w = compute_group_moments(simul_panel_new_NFFs_w, parameters_new_NFFs_w; burnin=500);

#======================================#
# BAPCPA (No FFs) -- incentive premium #
#======================================#
ι_λ_, w_λ_ = parameters_new.ι_λ, parameters_old.w_λ
crit_VP_new_NFFs_ι, parameters_new_NFFs_ι, variables_new_NFFs_ι, simul_panel_new_NFFs_ι, flag_new_NFFs_ι = optimal_multiplier_function(Ph=1.0 / 10.0, κ=991.0 / 33176.0, β=0.9550, η=0.2250, ψ=0.970, θ=1.0 / Inf, ζ=0.0162, ι_λ_=ι_λ_, w_λ_=w_λ_);
mnts_ag_new_NFFs_ι, mnts_e1_new_NFFs_ι, mnts_e2_new_NFFs_ι, mnts_e3_new_NFFs_ι = compute_group_moments(simul_panel_new_NFFs_ι, parameters_new_NFFs_ι; burnin=500);

# variables = create_variables(parameters_new);
# itp_cache = build_itp_cache(variables, parameters_new);
# crit_VP = solve_value_and_policy_functions!(variables, itp_cache, parameters_new; tol=1E-6, relax_V=1.0, relax_q=1.0, bellman_step=1)

# solve_value_and_policy_functions!(variables, itp_cache, parameters; tol=1E-6, relax=1.0, bellman_step=1);
# update_simul_itp_cache!(simul_itp_cache, variables, parameters);
# simulate_household_panel!(simul_panel, parameters, simul_itp_cache);
# compute_moments!(variables, parameters, simul_panel; burnin=500);

a_range = range(-3, 20, length=101)
# histogram(reshape(simul_panel.asset_state[1001:end, :], :, 1), bins=a_range, normalize=:pdf, color=:blue)
histogram(reshape(simul_panel_old.asset_state[end, :], :, 1), bins=a_range, normalize=:pdf, color=:blue)
histogram(reshape(simul_panel_new.asset_state[end, :], :, 1), bins=a_range, normalize=:pdf, color=:blue)

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

plot(parameters_old.a_grid_neg, variables_old.q[1:parameters_old.a_size_neg, :, 1])
plot!(parameters_new.a_grid_neg, variables_new.q[1:parameters_new.a_size_neg, :, 1], ls=:dash)

plot(parameters_old.a_grid_neg, variables_old.q[1:parameters_old.a_size_neg, :, end-1])
plot!(parameters_new.a_grid_neg, variables_new.q[1:parameters_new.a_size_neg, :, end-1], ls=:dash)

plot(parameters_old.a_grid_neg, variables_old.q[1:parameters_old.a_size_neg, :, end])
plot!(parameters_new.a_grid_neg, variables_new.q[1:parameters_new.a_size_neg, :, end], ls=:dash)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, 1] .* parameters.a_grid_neg)
plot!(variables.rbl_a[:, 1], variables.rbl_qa[:, 1], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, end-1] .* parameters.a_grid_neg)
plot!(variables.rbl_a[:, end-1], variables.rbl_qa[:, end-1], seriestype=:scatter)

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :, end] .* parameters.a_grid_neg)
plot!(variables.rbl_a[:, end], variables.rbl_qa[:, end], seriestype=:scatter)


plot(parameters.a_grid[parameters.a_ind_zero:parameters.a_ind_zero+185], variables.V[parameters.a_ind_zero:parameters.a_ind_zero+185, 2, 1, :])
plot(parameters.a_grid[parameters.a_ind_zero:parameters.a_ind_zero+180], variables.V[parameters.a_ind_zero:parameters.a_ind_zero+180, 2, 1, :], seriestype=:scatter)

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