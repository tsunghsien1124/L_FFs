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
using UnicodePlots
using MAT
using Distributions
using StatsPlots
using KernelDensity

# include("functions_expenditure.jl")
include("functions_preference.jl")

# find steady state by calibrating the lower state of preference shock
# para_targeted_ν(x) = para_func(; ν = x)
# solve_targeted_ν(x) = solve_func!(vars_func(para_targeted_ν(x)), para_targeted_ν(x))
# fzero(solve_targeted_ν, (0.9130, 0.9131))
# parameters = para_func(; ν = 0.9130231957950928, L = 10.000000000000481)
# variables = vars_func(parameters)
# solve_func!(variables, parameters)

para_targeted_r_f_FA(x) = para_func(; r_f = x)
solve_targeted_r_f_FA(x) = solve_func!(vars_func(para_targeted_r_f_FA(x)), para_targeted_r_f_FA(x))
r_f_FA = fzero(solve_targeted_r_f_FA, 0.024846234136912112)

parameters_FA = para_func(; r_f = r_f_FA)
variables_FA = vars_func(parameters_FA)
solve_func!(variables_FA, parameters_FA)

para_targeted_r_f_FA_low_z(x) = para_func(; r_f = x, z = 0.99)
solve_targeted_r_f_FA_low_z(x) = solve_func!(vars_func(para_targeted_r_f_FA_low_z(x)), para_targeted_r_f_FA_low_z(x))
r_f_FA_low_z = fzero(solve_targeted_r_f_FA_low_z, 0.024955878189469188)

parameters_FA_low_z = para_func(; r_f = r_f_FA_low_z, z = 0.99)
variables_FA_low_z = vars_func(parameters_FA_low_z)
solve_func!(variables_FA_low_z, parameters_FA_low_z)

para_targeted_r_f_NFA(x) = para_func(; r_f = x, L = 1E+100, r_bf = 0)
solve_targeted_r_f_NFA(x) = solve_func!(vars_func(para_targeted_r_f_NFA(x)), para_targeted_r_f_NFA(x))
r_f_NFA = fzero(solve_targeted_r_f_NFA, 0.02805071052538877)

parameters_NFA = para_func(; r_f = r_f_NFA, L = 1E+100, r_bf = 0)
variables_NFA = vars_func(parameters_NFA)
solve_func!(variables_NFA, parameters_NFA)

para_targeted_r_f_NFA_low_z(x) = para_func(; r_f = x, L = 1E+100, r_bf = 0, z = 0.99)
solve_targeted_r_f_NFA_low_z(x) = solve_func!(vars_func(para_targeted_r_f_NFA_low_z(x)), para_targeted_r_f_NFA_low_z(x))
r_f_NFA_low_z = fzero(solve_targeted_r_f_NFA_low_z, 0.02816665773604365)

parameters_NFA_low_z = para_func(; r_f = r_f_NFA_low_z, L = 1E+100, r_bf = 0, z = 0.99)
variables_NFA_low_z = vars_func(parameters_NFA_low_z)
solve_func!(variables_NFA_low_z, parameters_NFA_low_z)

# parameters_no_FA = para_func(; ν = 0.9130231957950928, r_bf = 0)
# variables_no_FA = vars_func(parameters_no_FA)
# solve_func!(variables_no_FA, parameters_no_FA)

# T = 200
# vars_matlab = matread("MIT_z_N.mat")
# z_guess = vars_matlab["z"][:,1]
# N_guess = vars_matlab["N"][:,1].*variables.A[3]
# T = 20
# z_guess = ones(T)
# N_guess = ones(T) .* variables.A[3]
# variables_MIT = vars_MIT_func(z_guess, N_guess, variables, parameters; T = T)
# solve_MIT_func!(variables_MIT, variables, parameters; T = T)

function plot_q_func(variables::mut_vars, parameters::NamedTuple)
    p_grid_ = round.(parameters.p_grid; digits = 4)
    a_grid_spline = collect(-1:0.04:0)
    a_size_spline = length(a_grid_spline)
    q_spline = zeros(a_size_spline, parameters.p_size)
    for p_ind in 1:parameters.p_size
        q_itp = Spline1D(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg,p_ind,1]; k = 1, bc = "extrapolate")
        for a_ind in 1:a_size_spline
            q_spline[a_ind,p_ind] = q_itp(a_grid_spline[a_ind])
        end
    end
    label_latex = reshape(latexstring.("\$",["p=$(p_grid_[i])" for i in 1:parameters.p_size],"\$"),1,:)
    plot(a_grid_spline, q_spline, lw = 2,
         xlabel = "\$a'\$", ylabel = "\$q(a',s)\$",
         xlims = (-1,0), ylims = (0,1),
         xticks = -1:0.2:0, yticks = 0:0.2:1,
         label = label_latex,
         legend = :bottomright, legendfont = font(9),
         theme = theme(:ggplot2))
end
plot_q_FA = plot_q_func(variables_FA, parameters_FA)
savefig(plot_q_FA, "plot_q_FA.pdf")
plot_q_NFA = plot_q_func(variables_NFA, parameters_NFA)
savefig(plot_q_NFA, "plot_q_NFA.pdf")

function plot_rbl_func(variables::mut_vars, parameters::NamedTuple)
    p_grid_ = round.(parameters.p_grid; digits = 4)
    a_grid_spline = collect(-1:0.04:0)
    a_size_spline = length(a_grid_spline)
    qa_spline = zeros(a_size_spline, parameters.p_size)
    for p_ind in 1:parameters.p_size
        qa_itp = Spline1D(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg,p_ind,3]; k = 1, bc = "extrapolate")
        for a_ind in 1:a_size_spline
            qa_spline[a_ind,p_ind] = qa_itp(a_grid_spline[a_ind])
        end
    end
    label_latex = reshape(latexstring.("\$",["p=$(p_grid_[i])" for i in 1:parameters.p_size],"\$"),1,:)
    plot(a_grid_spline, -qa_spline, lw = 2,
         xlabel = "\$a'\$", ylabel = "\$|q(a',s)a'|\$",
         xlims = (-1,0), ylims = (0,1.0),
         xticks = -1:0.2:0, yticks = 0:0.2:1.0,
         label = label_latex,
         legend = :topright, legendfont = font(9),
         theme = theme(:wong))
end
plot_rbl_func(variables_FA, parameters_FA)
plot_rbl_func(variables_NFA, parameters_NFA)

function plot_PD_func(variables::mut_vars, parameters::NamedTuple)
    lower_bound = -0.2
    a_ind_upper = maximum(findall(parameters.a_grid .<= lower_bound))
    PD = 1 .- variables.q[1:parameters.a_size_neg,1:parameters.p_size,1] .* (1+parameters.r_f+parameters.r_bf)
    plot(parameters.p_grid, parameters.a_grid_neg[1:a_ind_upper], PD[1:a_ind_upper,:],
         st = :heatmap, clim = (0,1), color = :dense,
         xlabel = "\$p\$", ylabel = "\$a'\$",
         yticks = -1.0:0.1:lower_bound,
         colorbar_title = "\$P(g'_h(a',s)=1)\$",
         theme = theme(:ggplot2))
end
plot_PD_FA = plot_PD_func(variables_FA, parameters_FA)
savefig(plot_PD_FA, "plot_PD_FA.pdf")
plot_PD_NFA = plot_PD_func(variables_NFA, parameters_NFA)
savefig(plot_PD_NFA, "plot_PD_NFA.pdf")

function table_func(variables::mut_vars, parameters::NamedTuple)
    results = zeros(9)
    # loans
    results[1] = variables.A[1]
    # deposits
    results[2] = variables.A[2]
    # net_worth
    results[3] = variables.A[3]
    # leverage_ratio
    results[4] = variables.A[4]
    # borrowing rate
    results[5] = parameters.r_f + parameters.r_bf
    # risk_free_rate
    results[6] = parameters.r_f
    # excess_return
    results[7] = parameters.r_bf
    # good_history
    results[8] = sum(variables.μ[1:parameters.x_size*parameters.a_size])
    # bad_history
    results[9] = sum(variables.μ[parameters.x_size*parameters.a_size+1:end])
    return results
end
table_FA = table_func(variables_FA, parameters_FA)
table_NFA = table_func(variables_NFA, parameters_NFA)
table_FA_low_z = table_func(variables_FA_low_z, parameters_FA_low_z)
table_NFA_low_z = table_func(variables_NFA_low_z, parameters_NFA_low_z)

table_compare_FA = round.(((table_FA .- table_NFA) ./ table_NFA) .* 100; digits = 2)
table_compare_FA_z = round.(((table_FA_low_z .- table_FA) ./ table_FA) .* 100; digits = 2)
table_compare_NFA_z = round.(((table_NFA_low_z .- table_NFA) ./ table_NFA) .* 100; digits = 2)

table_FA = round.(table_FA; digits = 4)
table_NFA = round.(table_NFA; digits = 4)
table_FA_low_z = round.(table_FA_low_z; digits = 4)
table_NFA_low_z = round.(table_NFA_low_z; digits = 4)

table_total_FA = [table_NFA table_FA table_compare_FA]
table_total_z = [table_NFA table_NFA_low_z table_compare_NFA_z table_FA table_FA_low_z table_compare_FA_z]

# Chekcing Plots, seriestype=:scatter
# (1) bond price (q)
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
plot(parameters.a_grid_neg, variables.V_good[1:parameters.a_size_neg,1:parameters.p_size,1])

plot(parameters.a_grid_pos,variables.policy_a_bad[:,1:parameters.p_size], legend = :bottomright)
plot(parameters.a_grid, variables.policy_a_good[:,1:parameters.p_size,1], legend = :bottomright)
plot(parameters.a_grid_neg, variables.policy_a_good[1:parameters.a_size_neg,1:parameters.p_size,1])
