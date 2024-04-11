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
# include("solving_transitional_dynamics.jl")
# include("simulation.jl")

#==============================#
# Solve stationary equilibrium #
#==============================#
parameters = parameters_function();
variables = variables_function(parameters; λ=0.0, load_init=true);
slow_updating = 1.0;
ED_KL_to_D_ratio_min, ED_leverage_ratio_min, crit_V_min, crit_μ_min = solve_economy_function!(variables, parameters; slow_updating=slow_updating);
V, V_d, V_nd, V_pos, R, q, rbl, μ = variables.V, variables.V_d, variables.V_nd, variables.V_pos, variables.R, variables.q, variables.rbl, variables.μ;
@save "results_int.jld2" V V_d V_nd V_pos R q rbl μ;

variables_λ_lower, variables, flag, crit_V, crit_μ = optimal_multiplier_function(parameters; slow_updating=slow_updating);