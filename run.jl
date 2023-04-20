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
parameters = parameters_function()
variables = variables_function(parameters; λ=0.0)

tol_h = 1E-8
slow_updating = 1.0
iter_max=500

crit_V = solve_value_and_pricing_function!(variables, parameters; tol = tol_h, iter_max = 500, slow_updating = slow_updating)

plot(parameters.a_grid_neg,variables.q[1:parameters.a_ind_zero,1,:])
plot(parameters.a_grid_neg,variables.V[1:parameters.a_ind_zero,2,3,:,1])
plot(parameters.e_2_grid, variables.threshold_a[1,:,:,2])
plot(parameters.a_grid_neg, variables.threshold_e_2[:,1,:,2])