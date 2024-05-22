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
variables = variables_function(parameters; λ=0.04244494091796878, load_init=false);
slow_updating = 1.0;
ED_KL_to_D_ratio_min, ED_leverage_ratio_min, crit_V_min, crit_μ_min = solve_economy_function!(variables, parameters; slow_updating=slow_updating);
V, V_d, V_nd, V_pos, R, q, rbl, μ = variables.V, variables.V_d, variables.V_nd, variables.V_pos, variables.R, variables.q, variables.rbl, variables.μ;
@save "results_int.jld2" V V_d V_nd V_pos R q rbl μ;
# variables_λ_lower, variables, flag, crit_V, crit_μ = optimal_multiplier_function(parameters; slow_updating=slow_updating);
# variables.aggregate_prices.λ

#================#
# Checking plots #
#================#
plot(parameters.a_grid_neg, variables.q[1:parameters.a_ind_zero,2,:], color=[:red :blue :black], label=:none)
plot!(parameters.a_grid_neg, variables.q[1:parameters.a_ind_zero,1,:], color=[:red :blue :black], label=:none, linestyle=:dash)

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
if isfile("C:/Users/User/Documents/Consumer_credit_FFs/results/jld2/transition_path_p_h.jld2")
    @load "C:/Users/User/Documents/Consumer_credit_FFs/results/jld2/transition_path_p_h.jld2" transition_path_p_h
    variables_T_p_h = variables_T_function(transition_path_p_h, variables_p_h_1, variables_p_h_2, parameters_p_h_2; T_size=T_size, T_degree=T_degree);
else
    variables_T_p_h = variables_T_function(variables_p_h_1, variables_p_h_2, parameters_p_h_2; T_size=T_size, T_degree=T_degree);
end
transitional_dynamic_λ_function!(variables_T_p_h, variables_p_h_1, variables_p_h_2, parameters_p_h_2; tol=tol, iter_max=iter_max, slow_updating=slow_updating_transitional_dynamics)
transition_path_p_h = variables_T_p_h.aggregate_prices.leverage_ratio_λ
@save "C:/Users/User/Documents/Consumer_credit_FFs/results/jld2/transition_path_p_h.jld2" transition_path_p_h
plot_transition_path_p_h = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_p_h = plot!(transition_path_p_h, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_p_h
savefig(plot_transition_path_p_h, "C:/Users/User/Documents/Consumer_credit_FFs/results/figures/plot_transition_path_p_h.pdf")

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
# λ_BAPCPA_1 = variables_BAPCPA_1.aggregate_prices.λ # 0.0279290036621094
variables_BAPCPA_1 = variables_function(parameters_BAPCPA_1; λ=0.0279290036621094);
ED_KL_to_D_ratio_min_BAPCPA_1, ED_leverage_ratio_min_BAPCPA_1, crit_V_min_BAPCPA_1, crit_μ_min_BAPCPA_1 = solve_economy_function!(variables_BAPCPA_1, parameters_BAPCPA_1; slow_updating=slow_updating);

# new economy - post BAPCPA
parameters_BAPCPA_2 = parameters_function(κ = κ_2, p_h = p_h_2);
# variables_λ_lower_BAPCPA_2, variables_BAPCPA_2, flag_BAPCPA_2, crit_V_BAPCPA_2, crit_μ_BAPCPA_2 = optimal_multiplier_function(parameters_BAPCPA_2; slow_updating=slow_updating);
# λ_BAPCPA_2 = variables_BAPCPA_2.aggregate_prices.λ # 0.03187119824218752
variables_BAPCPA_2 = variables_function(parameters_BAPCPA_2; λ=0.03187119824218752);
ED_KL_to_D_ratio_min_BAPCPA_2, ED_leverage_ratio_min_BAPCPA_2, crit_V_min_BAPCPA_2, crit_μ_min_BAPCPA_2 = solve_economy_function!(variables_BAPCPA_2, parameters_BAPCPA_2; slow_updating=slow_updating);

# set parameters for computation
T_size = 80
T_degree = 15.0
iter_max = 500
tol = 1E-4
slow_updating_transitional_dynamics = 0.1
initial_z = ones(T_size + 2);

# from pre to post BAPCPA
if isfile("C:/Users/User/Documents/Consumer_credit_FFs/results/jld2/transition_path_BAPCPA.jld2")
    @load "C:/Users/User/Documents/Consumer_credit_FFs/results/jld2/transition_path_BAPCPA.jld2" transition_path_BAPCPA
    variables_T_BAPCPA = variables_T_function(transition_path_BAPCPA, variables_BAPCPA_1, variables_BAPCPA_2, parameters_BAPCPA_2; T_size=T_size, T_degree=T_degree);

else
    variables_T_BAPCPA = variables_T_function(variables_BAPCPA_1, variables_BAPCPA_2, parameters_BAPCPA_2; T_size=T_size, T_degree=T_degree);
end
transitional_dynamic_λ_function!(variables_T_BAPCPA, variables_BAPCPA_1, variables_BAPCPA_2, parameters_BAPCPA_2; tol=tol, iter_max=iter_max, slow_updating=slow_updating_transitional_dynamics)
transition_path_BAPCPA = variables_T_BAPCPA.aggregate_prices.leverage_ratio_λ
@save "C:/Users/User/Documents/Consumer_credit_FFs/results/jld2/transition_path_BAPCPA.jld2" transition_path_BAPCPA
plot_transition_path_BAPCPA = plot(size=(800, 500), box=:on, legend=:bottomright, xtickfont=font(18, "Computer Modern", :black), ytickfont=font(18, "Computer Modern", :black), titlefont=font(18, "Computer Modern", :black), guidefont=font(18, "Computer Modern", :black), legendfont=font(18, "Computer Modern", :black), margin=4mm, ylabel="", xlabel="Period")
plot_transition_path_BAPCPA = plot!(transition_path_BAPCPA, linecolor=:blue, linewidth=3, markershapes=:circle, markercolor=:blue, markersize=6, markerstrokecolor=:blue, label=:none)
plot_transition_path_BAPCPA
savefig(plot_transition_path_BAPCPA, "C:/Users/User/Documents/Consumer_credit_FFs/results/figures/transition_path_BAPCPA.pdf")
