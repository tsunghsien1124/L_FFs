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
