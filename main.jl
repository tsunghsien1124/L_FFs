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

include("functions_preference.jl")

parameters = para()
variables = vars(parameters)

households!(variables, parameters)
banks!(variables, parameters)
