#=================#
# Import packages #
#=================#
using Dierckx
using FLOWMath
using Distributions
using QuadGK
using JLD2: @save, @load
using LinearAlgebra
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
using Random
using GLM
using DataFrames
using Measures
using BenchmarkTools, Profile
using Polyester
using Interpolations

#==================#
# Import functions #
#==================#
include("solving_stationary_equilibrium.jl")
include("solving_transitional_dynamics.jl")
# include("simulation.jl")

#==============================#
# Solve stationary equilibrium #
#==============================#
parameters = parameters_function();
variables = variables_function(parameters; λ=0.04244494091796878, load_init=false);
slow_updating = 1.0;
@btime crit_V = solve_value_and_pricing_function!(variables, parameters; tol=1E-6, iter_max=500, slow_updating=slow_updating);


@profview crit_V = solve_value_and_pricing_function!(variables, parameters; tol=1E-6, iter_max=500, slow_updating=slow_updating);

@time ED_KL_to_D_ratio_min, ED_leverage_ratio_min, crit_V_min, crit_μ_min = solve_economy_function!(variables, parameters; slow_updating=slow_updating);
V, V_d, V_nd, V_pos, R, q, rbl, μ = variables.V, variables.V_d, variables.V_nd, variables.V_pos, variables.R, variables.q, variables.rbl, variables.μ;
@save "results_int.jld2" V V_d V_nd V_pos R q rbl μ;
# variables_λ_lower, variables, flag, crit_V, crit_μ = optimal_multiplier_function(parameters; slow_updating=slow_updating);
# variables.aggregate_prices.λ

#================#
# Checking plots #
#================#
plot(parameters.a_grid_neg, variables.q[1:parameters.a_ind_zero,2,:], color=[:red :blue :black], label=:none)
plot!(parameters.a_grid_neg, variables.q[1:parameters.a_ind_zero,1,:], color=[:red :blue :black], label=:none, linestyle=:dash)

scatter(parameters.a_grid_neg[90:end], variables.q[90:parameters.a_ind_zero,2,1] .* parameters.a_grid_neg[90:end], color=[:red :blue :black], label=:none)
plot!(parameters.a_grid_neg, variables.q[1:parameters.a_ind_zero,1,:] .* parameters.a_grid_neg, color=[:red :blue :black], label=:none, linestyle=:dash)

#============================================#
# Solve transitional dynamics - Filing costs #
#============================================#
# cases
κ_1 = 697 / 33176;
κ_2 = 975 / 33176;
slow_updating = 1.0;

# old economy - low κ
parameters_κ_1 = parameters_function(κ = κ_1);
# variables_λ_lower_κ_1, variables_κ_1, flag_κ_1, crit_V_κ_1, crit_μ_κ_1 = optimal_multiplier_function(parameters_κ_1; slow_updating=slow_updating);
# λ_κ_1 = variables_κ_1.aggregate_prices.λ # 0.04244494091796878
variables_κ_1 = variables_function(parameters_κ_1; λ=0.04244494091796878, load_init=false);
ED_KL_to_D_ratio_min_κ_1, ED_leverage_ratio_min_κ_1, crit_V_min_κ_1, crit_μ_min_κ_1 = solve_economy_function!(variables_κ_1, parameters_κ_1; slow_updating=slow_updating);

# new economy - high κ
parameters_κ_2 = parameters_function(κ = κ_2);
# variables_λ_lower_κ_2, variables_κ_2, flag_κ_2, crit_V_κ_2, crit_μ_κ_2 = optimal_multiplier_function(parameters_κ_2; slow_updating=slow_updating);
# λ_κ_2 = variables_κ_2.aggregate_prices.λ # 0.02893077099609377
variables_κ_2 = variables_function(parameters_κ_2; λ=0.02893077099609377, load_init=false);
ED_KL_to_D_ratio_min_κ_2, ED_leverage_ratio_min_κ_2, crit_V_min_κ_2, crit_μ_min_κ_2 = solve_economy_function!(variables_κ_2, parameters_κ_2; slow_updating=slow_updating);

# set parameters for computation
T_size = 80;
T_degree = 15.0;
iter_max = 500;
tol = 1E-4;
slow_updating_transitional_dynamics = 0.1;
initial_z = ones(T_size + 2);

# from κ_1 to κ_2
if isfile(pwd() * "\\results\\jld2\\transition_path_κ.jld2")
    @load pwd() * "\\results\\jld2\\transition_path_κ.jld2" transition_path_κ
    variables_T_κ = variables_T_function(transition_path_κ, variables_κ_1, variables_κ_2, parameters_κ_2; T_size=T_size, T_degree=T_degree);
else
    variables_T_κ = variables_T_function(variables_κ_1, variables_κ_2, parameters_κ_2; T_size=T_size, T_degree=T_degree);
end
transitional_dynamic_λ_function!(variables_T_κ, variables_κ_1, variables_κ_2, parameters_κ_2; tol=tol, iter_max=iter_max, slow_updating=slow_updating_transitional_dynamics)
transition_path_κ = variables_T_κ.aggregate_prices.leverage_ratio_λ
@save pwd() * "\\results\\jld2\\transition_path_κ.jld2" transition_path_κ
plot_transition_path_κ = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_κ = plot!(transition_path_κ, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_κ
savefig(plot_transition_path_κ,  pwd() * "\\results\\figures\\plot_transition_path_κ.pdf")

transition_path_κ_N = variables_T_κ.aggregate_variables.N
plot_transition_path_κ_N = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_κ_N = plot!(transition_path_κ_N, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_κ_N
savefig(plot_transition_path_κ_N, pwd() * "\\results\\figures\\plot_transition_path_κ_N.pdf")

#=========================================#
# Solve transitional dynamics - Exclusion #
#=========================================#
# cases
p_h_1 = 1.0 / 6.0;
p_h_2 = 1.0 / 10.0;
slow_updating = 1.0;

# old economy - shorter p_h
parameters_p_h_1 = parameters_function(p_h = p_h_1);
# variables_λ_lower_p_h_1, variables_p_h_1, flag_p_h_1, crit_V_p_h_1, crit_μ_p_h_1 = optimal_multiplier_function(parameters_p_h_1; slow_updating=slow_updating);
# λ_p_h_1 = variables_p_h_1.aggregate_prices.λ # 0.04244494091796878
variables_p_h_1 = variables_function(parameters_p_h_1; λ=0.04244494091796878, load_init=false);
ED_KL_to_D_ratio_min_p_h_1, ED_leverage_ratio_min_p_h_1, crit_V_min_p_h_1, crit_μ_min_p_h_1 = solve_economy_function!(variables_p_h_1, parameters_p_h_1; slow_updating=slow_updating);

# new economy - longer p_h
parameters_p_h_2 = parameters_function(p_h = p_h_2);
# variables_λ_lower_p_h_2, variables_p_h_2, flag_p_h_2, crit_V_p_h_2, crit_μ_p_h_2 = optimal_multiplier_function(parameters_p_h_2; slow_updating=slow_updating);
# λ_p_h_2 = variables_p_h_2.aggregate_prices.λ # 0.04315687817382817
variables_p_h_2 = variables_function(parameters_p_h_2; λ=0.04315687817382817, load_init=false);
ED_KL_to_D_ratio_min_p_h_2, ED_leverage_ratio_min_p_h_2, crit_V_min_p_h_2, crit_μ_min_p_h_2 = solve_economy_function!(variables_p_h_2, parameters_p_h_2; slow_updating=slow_updating);

# set parameters for computation
T_size = 80
T_degree = 15.0
iter_max = 500
tol = 1E-4
slow_updating_transitional_dynamics = 0.1
initial_z = ones(T_size + 2);

# from p_h_1 to p_h_2
if isfile(pwd() * "\\results\\jld2\\transition_path_p_h.jld2")
    @load pwd() * "\\results\\jld2\\transition_path_p_h.jld2" transition_path_p_h
    variables_T_p_h = variables_T_function(transition_path_p_h, variables_p_h_1, variables_p_h_2, parameters_p_h_2; T_size=T_size, T_degree=T_degree);
else
    variables_T_p_h = variables_T_function(variables_p_h_1, variables_p_h_2, parameters_p_h_2; T_size=T_size, T_degree=T_degree);
end
transitional_dynamic_λ_function!(variables_T_p_h, variables_p_h_1, variables_p_h_2, parameters_p_h_2; tol=tol, iter_max=iter_max, slow_updating=slow_updating_transitional_dynamics)
transition_path_p_h = variables_T_p_h.aggregate_prices.leverage_ratio_λ
@save pwd() * "\\results\\jld2\\transition_path_p_h.jld2" transition_path_p_h
plot_transition_path_p_h = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_p_h = plot!(transition_path_p_h, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_p_h
savefig(plot_transition_path_p_h, pwd() * "\\results\\figures\\plot_transition_path_p_h.pdf")

#===========================================#
# Solve transitional dynamics - 2005 BAPCPA #
#===========================================#
# cases
κ_1, p_h_1 = 697 / 33176, 1.0 / 6.0;
κ_2, p_h_2 = 975 / 33176, 1.0 / 10.0;
slow_updating = 1.0;

# old economy - pre BAPCPA
parameters_BAPCPA_1 = parameters_function(κ = κ_1, p_h = p_h_1);
# variables_λ_lower_BAPCPA_1, variables_BAPCPA_1, flag_BAPCPA_1, crit_V_BAPCPA_1, crit_μ_BAPCPA_1 = optimal_multiplier_function(parameters_BAPCPA_1; slow_updating=slow_updating);
# λ_BAPCPA_1 = variables_BAPCPA_1.aggregate_prices.λ # 0.04244494091796878
variables_BAPCPA_1 = variables_function(parameters_BAPCPA_1; λ=0.04244494091796878, load_init=false);
ED_KL_to_D_ratio_min_BAPCPA_1, ED_leverage_ratio_min_BAPCPA_1, crit_V_min_BAPCPA_1, crit_μ_min_BAPCPA_1 = solve_economy_function!(variables_BAPCPA_1, parameters_BAPCPA_1; slow_updating=slow_updating);

# new economy - post BAPCPA
parameters_BAPCPA_2 = parameters_function(κ = κ_2, p_h = p_h_2);
# variables_λ_lower_BAPCPA_2, variables_BAPCPA_2, flag_BAPCPA_2, crit_V_BAPCPA_2, crit_μ_BAPCPA_2 = optimal_multiplier_function(parameters_BAPCPA_2; slow_updating=slow_updating);
# λ_BAPCPA_2 = variables_BAPCPA_2.aggregate_prices.λ # 0.03357268615722659
variables_BAPCPA_2 = variables_function(parameters_BAPCPA_2; λ=0.03357268615722659, load_init=false);
ED_KL_to_D_ratio_min_BAPCPA_2, ED_leverage_ratio_min_BAPCPA_2, crit_V_min_BAPCPA_2, crit_μ_min_BAPCPA_2 = solve_economy_function!(variables_BAPCPA_2, parameters_BAPCPA_2; slow_updating=slow_updating);

# new economy - post BAPCPA (low ν)
parameters_BAPCPA_2_low_ν = parameters_function(κ = κ_2, p_h = p_h_2, ν_size = 4);
# variables_λ_lower_BAPCPA_2_low_ν, variables_BAPCPA_2_low_ν, flag_BAPCPA_2_low_ν, crit_V_BAPCPA_2_low_ν, crit_μ_BAPCPA_2_low_ν = optimal_multiplier_function(parameters_BAPCPA_2_low_ν; slow_updating=slow_updating);
# λ_BAPCPA_2_low_ν = variables_BAPCPA_2_low_ν.aggregate_prices.λ # 0.03253384753417972
variables_BAPCPA_2_low_ν = variables_function(parameters_BAPCPA_2_low_ν; λ=0.03253384753417972, load_init=false);
ED_KL_to_D_ratio_min_BAPCPA_2_low_ν, ED_leverage_ratio_min_BAPCPA_2_low_ν, crit_V_min_BAPCPA_2_low_ν, crit_μ_min_BAPCPA_2_low_ν = solve_economy_function!(variables_BAPCPA_2_low_ν, parameters_BAPCPA_2_low_ν; slow_updating=slow_updating);

# # new economy - post BAPCPA (low ν and σ_ν)
# parameters_BAPCPA_2_low_ν_σ = parameters_function(κ = κ_2, p_h = p_h_2, ν_size = 6);
# # variables_λ_lower_BAPCPA_2_low_ν_σ, variables_BAPCPA_2_low_ν_σ, flag_BAPCPA_2_low_ν_σ, crit_V_BAPCPA_2_low_ν_σ, crit_μ_BAPCPA_2_low_ν_σ = optimal_multiplier_function(parameters_BAPCPA_2_low_ν_σ; slow_updating=slow_updating);
# # λ_BAPCPA_2_low_ν_σ = variables_BAPCPA_2_low_ν_σ.aggregate_prices.λ # 0.03253384753417972
# variables_BAPCPA_2_low_ν_σ = variables_function(parameters_BAPCPA_2_low_ν_σ; λ=0.0, load_init=false);
# ED_KL_to_D_ratio_min_BAPCPA_2_low_ν_σ, ED_leverage_ratio_min_BAPCPA_2_low_ν_σ, crit_V_min_BAPCPA_2_low_ν_σ, crit_μ_min_BAPCPA_2_low_ν_σ = solve_economy_function!(variables_BAPCPA_2_low_ν_σ, parameters_BAPCPA_2_low_ν_σ; slow_updating=slow_updating);

# new economy - post BAPCPA (no ν)
parameters_BAPCPA_2_no_ν = parameters_function(κ = κ_2, p_h = p_h_2, ν_size = 5);
# variables_λ_lower_BAPCPA_2_no_ν, variables_BAPCPA_2_no_ν, flag_BAPCPA_2_no_ν, crit_V_BAPCPA_2_no_ν, crit_μ_BAPCPA_2_no_ν = optimal_multiplier_function(parameters_BAPCPA_2_no_ν; slow_updating=slow_updating);
# λ_BAPCPA_2_no_ν = variables_BAPCPA_2_no_ν.aggregate_prices.λ # 0.03253384753417972
variables_BAPCPA_2_no_ν = variables_function(parameters_BAPCPA_2_no_ν; λ=0.0, load_init=false);
ED_KL_to_D_ratio_min_BAPCPA_2_no_ν, ED_leverage_ratio_min_BAPCPA_2_no_ν, crit_V_min_BAPCPA_2_no_ν, crit_μ_min_BAPCPA_2_no_ν = solve_economy_function!(variables_BAPCPA_2_no_ν, parameters_BAPCPA_2_no_ν; slow_updating=slow_updating);

# set parameters for computation
T_size = 80
T_degree = 15.0
iter_max = 500
tol = 1E-4
slow_updating_transitional_dynamics = 0.1
initial_z = ones(T_size + 2);

# from pre to post BAPCPA
if isfile(pwd() * "\\results\\jld2\\transition_path_BAPCPA.jld2")
    @load pwd() * "\\results\\jld2\\transition_path_BAPCPA.jld2" transition_path_BAPCPA
    variables_T_BAPCPA = variables_T_function(transition_path_BAPCPA, variables_BAPCPA_1, variables_BAPCPA_2, parameters_BAPCPA_2; T_size=T_size, T_degree=T_degree);
else
    variables_T_BAPCPA = variables_T_function(variables_BAPCPA_1, variables_BAPCPA_2, parameters_BAPCPA_2; T_size=T_size, T_degree=T_degree);
end
transitional_dynamic_λ_function!(variables_T_BAPCPA, variables_BAPCPA_1, variables_BAPCPA_2, parameters_BAPCPA_2; tol=tol, iter_max=iter_max, slow_updating=slow_updating_transitional_dynamics)
transition_path_BAPCPA = variables_T_BAPCPA.aggregate_prices.leverage_ratio_λ
@save pwd() * "\\results\\jld2\\transition_path_BAPCPA.jld2" transition_path_BAPCPA
plot_transition_path_BAPCPA = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_BAPCPA = plot!(transition_path_BAPCPA, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_BAPCPA
savefig(plot_transition_path_BAPCPA, pwd() * "\\results\\figures\\transition_path_BAPCPA.pdf")

transition_path_BAPCPA_N = variables_T_BAPCPA.aggregate_variables.N
plot_transition_path_BAPCPA_N = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_BAPCPA_N = plot!(transition_path_BAPCPA_N, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_BAPCPA_N
savefig(plot_transition_path_BAPCPA_N, pwd() * "\\results\\figures\\transition_path_BAPCPA_N.pdf")

# CEV welfare
welfare_CEV_BAPCPA_good_with_debt = 100 * sum(((variables_T_BAPCPA.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 2] ./ variables_BAPCPA_1.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1])
# W_old_good_with_debt = sum(variables_BAPCPA_1.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :] .* variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1])
# W_new_good_with_debt = sum(variables_T_BAPCPA.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 2] .* variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1])
# welfare_CEV_BAPCPA_good_with_debt = 100 * ((W_new_good_with_debt / W_old_good_with_debt)^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) - 1.0)

welfare_CEV_BAPCPA_good_with_savings = 100 * sum(((variables_T_BAPCPA.V[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 2] ./ variables_BAPCPA_1.V[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 1])
# W_old_good_without_debt = sum(variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :] .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 1])
# W_new_good_without_debt = sum(variables_T_BAPCPA.V[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2] .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 1])
# welfare_CEV_BAPCPA_good_without_debt = 100 * ((W_new_good_without_debt / W_old_good_without_debt)^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) - 1.0)

welfare_CEV_BAPCPA_good_without_asset = 100 * sum(((variables_T_BAPCPA.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 2] ./ variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1])

welfare_CEV_BAPCPA_good = 100 * sum(((variables_T_BAPCPA.V[:, :, :, :, :, 2] ./ variables_BAPCPA_1.V).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[:, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[:, :, :, :, :, 1])
# W_old_good = W_old_good_with_debt + W_old_good_without_debt
# W_new_good = W_new_good_with_debt + W_new_good_without_debt
# welfare_CEV_BAPCPA_good = 100 * ((W_new_good / W_old_good)^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) - 1.0)

welfare_CEV_BAPCPA_bad = 100 * sum(((variables_T_BAPCPA.V_pos[:, :, :, :, :, 2] ./ variables_BAPCPA_1.V_pos).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2]) / sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2])
# W_old_bad = sum(variables_BAPCPA_1.V_pos[:, :, :, :, :] .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2])
# W_new_bad = sum(variables_T_BAPCPA.V_pos[:, :, :, :, :, 2] .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2])
# welfare_CEV_BAPCPA_bad = 100 * ((W_new_bad / W_old_bad)^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) - 1.0)

welfare_CEV_BAPCPA = 100 * (sum(((variables_T_BAPCPA.V[:, :, :, :, :, 2] ./ variables_BAPCPA_1.V).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[:, :, :, :, :, 1]) + sum(((variables_T_BAPCPA.V_pos[:, :, :, :, :, 2] ./ variables_BAPCPA_1.V_pos).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2]))
# W_old = W_old_good + W_old_bad
# W_new = W_new_good + W_new_bad
# welfare_CEV_BAPCPA = 100 * ((W_new / W_old)^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) - 1.0)

# favor share
welfare_favor_BAPCPA_good_with_debt = 100 * sum((variables_T_BAPCPA.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 2] .> variables_BAPCPA_1.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :]) .* (variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1] ./ sum(variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1])))

welfare_favor_BAPCPA_good_with_savings = 100 * sum((variables_T_BAPCPA.V[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 2] .> variables_BAPCPA_1.V[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :]) .* (variables_BAPCPA_1.μ[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 1] ./ sum(variables_BAPCPA_1.μ[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 1])))

welfare_favor_BAPCPA_good_without_asset = 100 * sum((variables_T_BAPCPA.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 2] .> variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :]) .* (variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1] ./ sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1])))

welfare_favor_BAPCPA_good = 100 * sum((variables_T_BAPCPA.V[:, :, :, :, :, 2] .> variables_BAPCPA_1.V) .* (variables_BAPCPA_1.μ[:, :, :, :, :, 1] ./ sum(variables_BAPCPA_1.μ[:, :, :, :, :, 1])))

welfare_favor_BAPCPA_bad = 100 * sum((variables_T_BAPCPA.V_pos[:, :, :, :, :, 2] .> variables_BAPCPA_1.V_pos) .* (variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2] ./ sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2])))

welfare_favor_BAPCPA = 100 * (sum((variables_T_BAPCPA.V[:, :, :, :, :, 2] .> variables_BAPCPA_1.V) .* variables_BAPCPA_1.μ[:, :, :, :, :, 1]) + sum((variables_T_BAPCPA.V_pos[:, :, :, :, :, 2] .> variables_BAPCPA_1.V_pos) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2]))

# from pre to post BAPCPA (low ν)
if isfile(pwd() * "\\results\\jld2\\transition_path_BAPCPA_low_ν.jld2")
    @load pwd() * "\\results\\jld2\\transition_path_BAPCPA_low_ν.jld2" transition_path_BAPCPA_low_ν
    variables_T_BAPCPA_low_ν = variables_T_function(transition_path_BAPCPA_low_ν, variables_BAPCPA_1, variables_BAPCPA_2_low_ν, parameters_BAPCPA_2_low_ν; T_size=T_size, T_degree=T_degree);
else
    variables_T_BAPCPA_low_ν = variables_T_function(variables_BAPCPA_1, variables_BAPCPA_2_low_ν, parameters_BAPCPA_2_low_ν; T_size=T_size, T_degree=T_degree);
end
transitional_dynamic_λ_function!(variables_T_BAPCPA_low_ν, variables_BAPCPA_1, variables_BAPCPA_2_low_ν, parameters_BAPCPA_2_low_ν; tol=tol, iter_max=iter_max, slow_updating=slow_updating_transitional_dynamics)
transition_path_BAPCPA_low_ν = variables_T_BAPCPA_low_ν.aggregate_prices.leverage_ratio_λ
@save pwd() * "\\results\\jld2\\transition_path_BAPCPA_low_ν.jld2" transition_path_BAPCPA_low_ν
plot_transition_path_BAPCPA_low_ν = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_BAPCPA_low_ν = plot!(transition_path_BAPCPA_low_ν, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_BAPCPA_low_ν
savefig(plot_transition_path_BAPCPA_low_ν, pwd() * "\\results\\figures\\transition_path_BAPCPA_low_ν.pdf")

transition_path_BAPCPA_low_ν_N = variables_T_BAPCPA_low_ν.aggregate_variables.N
plot_transition_path_BAPCPA_low_ν_N = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_BAPCPA_low_ν_N = plot!(transition_path_BAPCPA_low_ν_N, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_BAPCPA_low_ν_N
savefig(plot_transition_path_BAPCPA_low_ν_N, pwd() * "\\results\\figures\\transition_path_BAPCPA_low_ν_N.pdf")

# figure comparison
plot_transition_path_BAPCPA_comparison = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_BAPCPA_comparison = plot!(transition_path_BAPCPA, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_BAPCPA_comparison = plot!(transition_path_BAPCPA_low_ν, linecolor=:red, linewidth=3, markershapes=:circle, markercolor=:red, markersize=6, markerstrokecolor=:red, label=:none)
plot_transition_path_BAPCPA_comparison
savefig(plot_transition_path_BAPCPA_comparison, pwd() * "\\results\\figures\\plot_transition_path_BAPCPA_comparison.pdf")

plot_transition_path_BAPCPA_comparison_N = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_BAPCPA_comparison_N = plot!(transition_path_BAPCPA_N, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_BAPCPA_comparison_N = plot!(transition_path_BAPCPA_low_ν_N, linecolor=:red, linewidth=3, markershapes=:circle, markercolor=:red, markersize=6, markerstrokecolor=:red, label=:none)
plot_transition_path_BAPCPA_comparison_N
savefig(plot_transition_path_BAPCPA_comparison_N, pwd() * "\\results\\figures\\plot_transition_path_BAPCPA_comparison_N.pdf")

# CEV welfare
welfare_CEV_BAPCPA_low_ν_good_with_debt = 100 * sum(((variables_T_BAPCPA_low_ν.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 2] ./ variables_BAPCPA_1.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1])
# W_old_good_with_debt = sum(variables_BAPCPA_1.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :] .* variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1])
# W_new_good_with_debt = sum(variables_T_BAPCPA_low_ν.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 2] .* variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1])
# welfare_CEV_BAPCPA_low_ν_good_with_debt = 100 * ((W_new_good_with_debt / W_old_good_with_debt)^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) - 1.0)

welfare_CEV_BAPCPA_low_ν_good_with_savings = 100 * sum(((variables_T_BAPCPA_low_ν.V[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 2] ./ variables_BAPCPA_1.V[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 1])
# W_old_good_without_debt = sum(variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :] .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 1])
# W_new_good_without_debt = sum(variables_T_BAPCPA_low_ν.V[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2] .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 1])
# welfare_CEV_BAPCPA_low_ν_good_without_debt = 100 * ((W_new_good_without_debt / W_old_good_without_debt)^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) - 1.0)

welfare_CEV_BAPCPA_low_ν_good_without_asset = 100 * sum(((variables_T_BAPCPA_low_ν.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 2] ./ variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1])

welfare_CEV_BAPCPA_low_ν_good = 100 * sum(((variables_T_BAPCPA_low_ν.V[:, :, :, :, :, 2] ./ variables_BAPCPA_1.V).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[:, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[:, :, :, :, :, 1])
# W_old_good = W_old_good_with_debt + W_old_good_without_debt
# W_new_good = W_new_good_with_debt + W_new_good_without_debt
# welfare_CEV_BAPCPA_low_ν_good = 100 * ((W_new_good / W_old_good)^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) - 1.0)

welfare_CEV_BAPCPA_low_ν_bad = 100 * sum(((variables_T_BAPCPA_low_ν.V_pos[:, :, :, :, :, 2] ./ variables_BAPCPA_1.V_pos).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2]) / sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2])
# W_old_bad = sum(variables_BAPCPA_1.V_pos[:, :, :, :, :] .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2])
# W_new_bad = sum(variables_T_BAPCPA_low_ν.V_pos[:, :, :, :, :, 2] .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2])
# welfare_CEV_BAPCPA_low_ν_bad = 100 * ((W_new_bad / W_old_bad)^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) - 1.0)

welfare_CEV_BAPCPA_low_ν = 100 * (sum(((variables_T_BAPCPA_low_ν.V[:, :, :, :, :, 2] ./ variables_BAPCPA_1.V).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[:, :, :, :, :, 1]) + sum(((variables_T_BAPCPA_low_ν.V_pos[:, :, :, :, :, 2] ./ variables_BAPCPA_1.V_pos).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2]))
# W_old = W_old_good + W_old_bad
# W_new = W_new_good + W_new_bad
# welfare_CEV_BAPCPA_low_ν = 100 * ((W_new / W_old)^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) - 1.0)

# favor share
welfare_favor_BAPCPA_low_ν_good_with_debt = 100 * sum((variables_T_BAPCPA_low_ν.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 2] .> variables_BAPCPA_1.V[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :]) .* (variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1] ./ sum(variables_BAPCPA_1.μ[1:(parameters_BAPCPA_1.a_ind_zero-1), :, :, :, :, 1])))

welfare_favor_BAPCPA_low_ν_good_with_savings = 100 * sum((variables_T_BAPCPA_low_ν.V[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 2] .> variables_BAPCPA_1.V[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :]) .* (variables_BAPCPA_1.μ[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 1] ./ sum(variables_BAPCPA_1.μ[(parameters_BAPCPA_1.a_ind_zero+1):end, :, :, :, :, 1])))

welfare_favor_BAPCPA_low_ν_good_without_asset = 100 * sum((variables_T_BAPCPA_low_ν.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 2] .> variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :]) .* (variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1] ./ sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1])))

welfare_favor_BAPCPA_low_ν_good = 100 * sum((variables_T_BAPCPA_low_ν.V[:, :, :, :, :, 2] .> variables_BAPCPA_1.V) .* (variables_BAPCPA_1.μ[:, :, :, :, :, 1] ./ sum(variables_BAPCPA_1.μ[:, :, :, :, :, 1])))

welfare_favor_BAPCPA_low_ν_bad = 100 * sum((variables_T_BAPCPA_low_ν.V_pos[:, :, :, :, :, 2] .> variables_BAPCPA_1.V_pos) .* (variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2] ./ sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2])))

welfare_favor_BAPCPA_low_ν = 100 * (sum((variables_T_BAPCPA_low_ν.V[:, :, :, :, :, 2] .> variables_BAPCPA_1.V) .* variables_BAPCPA_1.μ[:, :, :, :, :, 1]) + sum((variables_T_BAPCPA_low_ν.V_pos[:, :, :, :, :, 2] .> variables_BAPCPA_1.V_pos) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2]))

# from pre to post BAPCPA (no ν)
welfare_CEV_BAPCPA_no_ν = 100 * (sum(((variables_BAPCPA_2_no_ν.V ./ variables_BAPCPA_1.V).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[:, :, :, :, :, 1]) + sum(((variables_BAPCPA_2_no_ν.V_pos ./ variables_BAPCPA_1.V_pos).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero:end, :, :, :, :, 2]))

#===============#
# Decomposition #
#===============#

# no financial frictions
variables_BAPCPA_2_NFF = variables_function(parameters_BAPCPA_2; λ=0.03357268615722659, load_init=false);
variables_BAPCPA_2_NFF.aggregate_prices.w_λ = variables_BAPCPA_1.aggregate_prices.w_λ;
variables_BAPCPA_2_NFF.aggregate_prices.ι_λ = variables_BAPCPA_1.aggregate_prices.ι_λ;
ED_KL_to_D_ratio_min_BAPCPA_2_NFF, ED_leverage_ratio_min_BAPCPA_2_NFF, crit_V_min_BAPCPA_2_NFF, crit_μ_min_BAPCPA_2_NFF = solve_economy_function!(variables_BAPCPA_2_NFF, parameters_BAPCPA_2; slow_updating=slow_updating);

# fixed wage
variables_BAPCPA_2_NFF_w = variables_function(parameters_BAPCPA_2; λ=0.03357268615722659, load_init=false);
variables_BAPCPA_2_NFF_w.aggregate_prices.w_λ = variables_BAPCPA_1.aggregate_prices.w_λ;
ED_KL_to_D_ratio_min_BAPCPA_2_NFF_w, ED_leverage_ratio_min_BAPCPA_2_NFF_w, crit_V_min_BAPCPA_2_NFF_w, crit_μ_min_BAPCPA_2_NFF_w = solve_economy_function!(variables_BAPCPA_2_NFF_w, parameters_BAPCPA_2; slow_updating=slow_updating);

# fixed iota
variables_BAPCPA_2_NFF_ι = variables_function(parameters_BAPCPA_2; λ=0.03357268615722659, load_init=false);
variables_BAPCPA_2_NFF_ι.aggregate_prices.ι_λ = variables_BAPCPA_1.aggregate_prices.ι_λ;
ED_KL_to_D_ratio_min_BAPCPA_2_NFF_ι, ED_leverage_ratio_min_BAPCPA_2_NFF_ι, crit_V_min_BAPCPA_2_NFF_ι, crit_μ_min_BAPCPA_2_NFF_ι = solve_economy_function!(variables_BAPCPA_2_NFF_ι, parameters_BAPCPA_2; slow_updating=slow_updating);

# CEV welfare
welfare_CEV_BAPCPA_newborn = 100 * sum(((variables_BAPCPA_2.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :] ./ variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1])

welfare_CEV_BAPCPA_low_ν_newborn = 100 * sum(((variables_BAPCPA_2_low_ν.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :] ./ variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1])

welfare_CEV_BAPCPA_NFF_newborn = 100 * sum(((variables_BAPCPA_2_NFF.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :] ./ variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1])

welfare_CEV_BAPCPA_NFF_w_newborn = 100 * sum(((variables_BAPCPA_2_NFF_w.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :] ./ variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1])

welfare_CEV_BAPCPA_NFF_ι_newborn = 100 * sum(((variables_BAPCPA_2_NFF_ι.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :] ./ variables_BAPCPA_1.V[parameters_BAPCPA_1.a_ind_zero, :, :, :, :]).^(1.0 / (1.0 - parameters_BAPCPA_1.σ)) .- 1.0) .* variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1]) / sum(variables_BAPCPA_1.μ[parameters_BAPCPA_1.a_ind_zero, :, :, :, :, 1])