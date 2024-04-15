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
variables = variables_function(parameters; λ=0.028143848632812524, load_init=true);
slow_updating = 1.0;
ED_KL_to_D_ratio_min, ED_leverage_ratio_min, crit_V_min, crit_μ_min = solve_economy_function!(variables, parameters; slow_updating=slow_updating);
V, V_d, V_nd, V_pos, R, q, rbl, μ = variables.V, variables.V_d, variables.V_nd, variables.V_pos, variables.R, variables.q, variables.rbl, variables.μ;
@save "results_int.jld2" V V_d V_nd V_pos R q rbl μ;
# variables_λ_lower, variables, flag, crit_V, crit_μ = optimal_multiplier_function(parameters; slow_updating=slow_updating);

#=============================#
# Solve transitional dynamics #
#=============================#
# cases
κ_1 = 697 / 33176
κ_2 = 975 / 33176
slow_updating = 1.0;

# old economy
parameters_κ_1 = parameters_function(κ = κ_1);
# variables_λ_lower_κ_1, variables_κ_1, flag_κ_1, crit_V_κ_1, crit_μ_κ_1 = optimal_multiplier_function(parameters_κ_1; slow_updating=slow_updating);
# λ_κ_1 = variables_κ_1.aggregate_prices.λ # 0.0279290036621094
variables_κ_1 = variables_function(parameters_κ_1; λ=0.0279290036621094);
ED_KL_to_D_ratio_min_κ_1, ED_leverage_ratio_min_κ_1, crit_V_min_κ_1, crit_μ_min_κ_1 = solve_economy_function!(variables_κ_1, parameters_κ_1; slow_updating=slow_updating);

# new economy
parameters_κ_2 = parameters_function(κ = κ_2);
# variables_λ_lower_κ_2, variables_κ_2, flag_κ_2, crit_V_κ_2, crit_μ_κ_2 = optimal_multiplier_function(parameters_κ_2; slow_updating=slow_updating);
# λ_κ_2 = variables_κ_2.aggregate_prices.λ # 0.026877527099609392
variables_κ_2 = variables_function(parameters_κ_2; λ=0.026877527099609392);
ED_KL_to_D_ratio_min_κ_2, ED_leverage_ratio_min_κ_2, crit_V_min_κ_2, crit_μ_min_κ_2 = solve_economy_function!(variables_κ_2, parameters_κ_2; slow_updating=slow_updating);

# set parameters for computation
T_size = 250
T_degree = 15.0
iter_max = 1
tol = 1E-2
slow_updating_transitional_dynamics = 0.5
initial_z = ones(T_size + 2);

# from κ_1 to κ_2
variables_T_κ = variables_T_function(variables_κ_1, variables_κ_2, parameters_κ_2; T_size=T_size, T_degree=T_degree);
transitional_dynamic_λ_function!(variables_T_κ, variables_κ_1, variables_κ_2, parameters_κ_2; tol=tol, iter_max=iter_max, slow_updating=slow_updating_transitional_dynamics)
transition_path_κ = variables_T_κ.aggregate_prices.leverage_ratio_λ
plot_transition_path_κ = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Time")
plot_transition_path_κ = plot!(transition_path_κ, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_κ