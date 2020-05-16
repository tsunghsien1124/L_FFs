#------------------------------------------------------------------------------#
#                           Import Necessary Packages                          #
#------------------------------------------------------------------------------#
using LinearAlgebra
using ProgressMeter
using Parameters
using QuantEcon
using Plots
using Optim
using Interpolations
using SparseArrays
using Roots

include("functions_preference.jl")
# include("functions_expenditure.jl")

parameters = para()
variables = vars(parameters)

solution!(variables, parameters)



plot(parameters.a_grid_neg,variables.q[1:parameters.a_size_neg,1:parameters.p_size])
plot(parameters.a_grid_pos,variables.V_bad)
plot(parameters.a_grid,variables.V_good)
plot(parameters.a_grid_neg,variables.V_good[1:parameters.a_size_neg,1:parameters.p_size],legend=:bottomright)
plot(parameters.a_grid_neg,-variables.q[1:parameters.a_size_neg,1:parameters.p_size].*parameters.a_grid_neg)
