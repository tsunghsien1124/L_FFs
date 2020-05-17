#------------------------------------------------------------------------------#
#                           Import Necessary Packages                          #
#------------------------------------------------------------------------------#
using LinearAlgebra
using ProgressMeter
using Parameters
using QuantEcon
using Plots
using PlotThemes
using Optim
using Interpolations
using SparseArrays
using Roots
using LaTeXStrings
using Dierckx

include("functions_preference.jl")
# include("functions_expenditure.jl")

parameters = para()
variables = vars(parameters)

solution!(variables, parameters)

# Chekcing Plots, seriestype=:scatter
# (1) bond price (q)
p_grid_ = round.(parameters.p_grid; digits = 4)
a_grid_spline = collect(-1:0.04:0)
a_size_spline = length(a_grid_spline)
q_spline = zeros(a_size_spline, parameters.p_size)
qa_spline = zeros(a_size_spline, parameters.p_size)
for p_ind in 1:parameters.p_size
    q_itp = Spline1D(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg,p_ind,1]; k = 1, bc = "extrapolate")
    qa_itp = Spline1D(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg,p_ind,3]; k = 1, bc = "extrapolate")
    for a_ind in 1:a_size_spline
        q_spline[a_ind,p_ind] = q_itp(a_grid_spline[a_ind])
        qa_spline[a_ind,p_ind] = qa_itp(a_grid_spline[a_ind])
    end
end
label_latex = reshape(latexstring.("\$",["p=$(p_grid_[i])" for i in 1:parameters.p_size],"\$"),1,:)
plot(a_grid_spline, q_spline, lw = 2,
     xlabel = "\$a'\$", ylabel = "\$q(a',s)\$",
     xlims = (-1,0), ylims = (0,1),
     xticks = -1:0.2:0, yticks = 0:0.2:1,
     label = label_latex,
     legend = :bottomright, legendfont = font(10), theme = theme(:wong))

plot(a_grid_spline, qa_spline, lw = 2,
     xlabel = "\$a'\$", ylabel = "\$q(a',s)a'\$",
     xlims = (-1,0), ylims = (-0.5,0),
     xticks = -1:0.2:0, yticks = -0.5:0.1:0,
     label = label_latex,
     legend = :bottomright, legendfont = font(10), theme = theme(:wong))

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg,1:parameters.p_size,1])
plot(parameters.a_grid, variables.q[:,1:parameters.p_size,1])

# (2) derivative of bond price (Dq)
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg,1:parameters.p_size,2])
plot(parameters.a_grid, variables.q[:,1:parameters.p_size,2], seriestype=:scatter)

# (3) size of bond (qa)
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg,1:parameters.p_size,3])
plot(parameters.a_grid, variables.q[:,1:parameters.p_size,3], seriestype=:scatter, legend=:bottomright)

# (4) derivative of size of bond (Dqa)
plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg,1:parameters.p_size,4], seriestype=:scatter)
plot(parameters.a_grid, variables.q[:,1:parameters.p_size,4], seriestype=:scatter, legend=:bottomright)

# (5) value functions
plot(parameters.a_grid_pos,variables.V_bad[:,1:parameters.p_size], legend = :bottomright)
plot(parameters.a_grid, variables.V_good[:,1:parameters.p_size,1], legend = :bottomright)
plot(parameters.a_grid_neg, variables.V_good[1:parameters.a_size_neg,1:parameters.p_size,1], seriestype=:scatter)
