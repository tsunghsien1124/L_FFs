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

#==================#
# Import functions #
#==================#
include("solving_stationary_equilibrium.jl")
# include("solving_stationary_equilibrium_fixed_cost.jl")
include("solving_transitional_dynamics.jl")
include("simulation.jl")

#===================#
# working directory #
#===================#
Indicator_local_machine = true
if Indicator_local_machine == true
    cd(homedir() * "\\Dropbox\\Dissertation\\Chapter 3 - Consumer Bankruptcy with Financial Frictions\\")
    # cd(homedir() * "/Dropbox/Dissertation/Chapter 3 - Consumer Bankruptcy with Financial Frictions/")
else
    cd(homedir() * "/financial_frictions/")
end

#=======#
# Tasks #
#=======#
Indicator_solve_equlibria_λ_min_and_max = false
Indicator_solve_equlibrium_given_λ = false
Indicator_solve_stationary_equlibrium = false
Indicator_solve_stationary_equlibria_across_η = false
Indicator_solve_transitional_dynamics = true
Indicator_simulation = false
Indicator_simulation_results = false

# print out the number of threads
println("Julia is running with $(Threads.nthreads()) threads...")
slow_updating = 1.0

#==============================#
# Solve stationary equilibrium #
#==============================#

if Indicator_solve_equlibria_λ_min_and_max == true

    parameters = parameters_function()

    variables_min = variables_function(parameters; λ = 0.0)
    ED_KL_to_D_ratio_min, ED_leverage_ratio_min = solve_economy_function!(variables_min, parameters; slow_updating = slow_updating)

    variables_max = variables_function(parameters; λ = 1.0 - sqrt(parameters.ψ))
    ED_KL_to_D_ratio_max, ED_leverage_ratio_max = solve_economy_function!(variables_max, parameters; slow_updating = slow_updating)

    data_spec = Any[
        "λ" variables_min.aggregate_prices.λ variables_max.aggregate_prices.λ
        "(K'+L')/D' (supply)" variables_min.aggregate_prices.KL_to_D_ratio_λ variables_max.aggregate_prices.KL_to_D_ratio_λ
        "(K'+L')/D' (demand)" variables_min.aggregate_variables.KL_to_D_ratio variables_max.aggregate_variables.KL_to_D_ratio
        "(K'+L')/N (supply)" variables_min.aggregate_prices.leverage_ratio_λ variables_max.aggregate_prices.leverage_ratio_λ
        "(K'+L')/N (demand)" variables_min.aggregate_variables.leverage_ratio variables_max.aggregate_variables.leverage_ratio
        "share of filers (%)" variables_min.aggregate_variables.share_of_filers*100 variables_max.aggregate_variables.share_of_filers*100
        "share in debts (%)" variables_min.aggregate_variables.share_in_debts*100 variables_max.aggregate_variables.share_in_debts*100
        "debt-to-earnings ratio (%)" variables_min.aggregate_variables.debt_to_earning_ratio*100 variables_max.aggregate_variables.debt_to_earning_ratio*100
        "avg interest rate (%)" variables_min.aggregate_variables.avg_loan_rate*100 variables_max.aggregate_variables.avg_loan_rate*100
        "policy upper bound" variables_min.policy_a[end, end, end, end, 2]<parameters.a_grid[end] variables_max.policy_a[end, end, end, end, 2]<parameters.a_grid[end]
    ]

    pretty_table(data_spec; header = ["Name", "λ minimum", "λ maximum"], alignment = [:c, :c, :c], formatters = ft_round(8), body_hlines = [3,5,9])
end

if Indicator_solve_equlibrium_given_λ == true

    parameters = parameters_function()
    variables = variables_function(parameters; λ = 0.0)
    # variables = variables_function(parameters; λ = 0.0169101590194511)
    ED_KL_to_D_ratio, ED_leverage_ratio = solve_economy_function!(variables, parameters; slow_updating = slow_updating)
    flag = 3

    calibration_results = [
        parameters.β,
        parameters.δ,
        parameters.ν_s,
        parameters.τ,
        parameters.p_h,
        parameters.η,
        parameters.θ,
        parameters.ν_p,
        variables.aggregate_prices.λ,
        variables.aggregate_variables.KL_to_D_ratio,
        variables.aggregate_variables.share_of_filers * 100,
        variables.aggregate_variables.D / variables.aggregate_variables.L,
        variables.aggregate_variables.share_in_debts * 100,
        variables.aggregate_variables.debt_to_earning_ratio * 100,
        variables.aggregate_variables.avg_loan_rate * 100,
        flag
        ]
        display(calibration_results)

end

#================#
# Checking Plots #
#================#
# a_neg_index = 1
# plot(parameters.a_grid_neg[a_neg_index:end], variables_min.q[a_neg_index:parameters.a_ind_zero,1,:], legend=:none)

# plot(parameters.a_grid_neg[a_neg_index:end], variables_min.policy_d[a_neg_index:parameters.a_ind_zero,2,:,1,2], legend=:none)

#============================================#
# Solve stationary equilibrium (calibration) #
#============================================#
if Indicator_solve_stationary_equlibrium == true

    β_search = 0.940 / 0.980 # collect(0.94:0.01:0.97)
    θ_search = 1.0 / 3.0 # eps() # collect(0.04:0.001:0.07)
    η_search = 0.25 # collect(0.20:0.05:0.40)
    ζ_d_search = 0.03000 # collect(0.03000:0.00100:0.03100)
    ν_p_search = 0.01018 # collect(0.010202:0.000001:0.010204)

    β_search_size = length(β_search)
    θ_search_size = length(θ_search)
    η_search_size = length(η_search)
    ζ_d_search_size = length(ζ_d_search)
    ν_p_search_size = length(ν_p_search)
    search_size = β_search_size * θ_search_size * η_search_size * ζ_d_search_size * ν_p_search_size
    calibration_results = zeros(search_size, 19)

    for β_i in 1:β_search_size, θ_i in 1:θ_search_size, η_i in 1:η_search_size, ζ_d_i in 1:ζ_d_search_size, ν_p_i in 1:ν_p_search_size

        parameters = parameters_function(β = β_search[β_i], θ = θ_search[θ_i], η = η_search[η_i], ζ_d = ζ_d_search[ζ_d_i], ν_p = ν_p_search[ν_p_i])
        variables_λ_lower, variables, flag = optimal_multiplier_function(parameters; slow_updating = slow_updating)

        search_iter = (β_i - 1)*(θ_search_size*η_search_size*ζ_d_search_size*ν_p_search_size) + (θ_i-1)*(η_search_size*ζ_d_search_size*ν_p_search_size) + (η_i-1)*ζ_d_search_size*ν_p_search_size + (ζ_d_i-1)*ν_p_search_size + ν_p_i

        calibration_results[search_iter, 1] = parameters.β
        calibration_results[search_iter, 2] = parameters.δ
        calibration_results[search_iter, 3] = parameters.τ
        calibration_results[search_iter, 4] = parameters.p_h
        calibration_results[search_iter, 5] = parameters.η
        calibration_results[search_iter, 6] = parameters.ψ
        calibration_results[search_iter, 7] = parameters.θ
        calibration_results[search_iter, 8] = parameters.ζ_d
        calibration_results[search_iter, 9] = parameters.ν_s
        calibration_results[search_iter, 10] = parameters.ν_p
        calibration_results[search_iter, 11] = variables.aggregate_prices.λ
        calibration_results[search_iter, 12] = variables.aggregate_variables.leverage_ratio
        calibration_results[search_iter, 13] = variables.aggregate_variables.KL_to_D_ratio
        calibration_results[search_iter, 14] = variables.aggregate_variables.share_of_filers * 100
        calibration_results[search_iter, 15] = variables.aggregate_variables.D / variables.aggregate_variables.L
        calibration_results[search_iter, 16] = variables.aggregate_variables.share_in_debts * 100
        calibration_results[search_iter, 17] = variables.aggregate_variables.debt_to_earning_ratio * 100
        calibration_results[search_iter, 18] = variables.aggregate_variables.avg_loan_rate * 100
        calibration_results[search_iter, 19] = flag
    end

    CSV.write("calibration_julia.csv", Tables.table(calibration_results), writeheader=false)

end

#======================================================#
# Solve the model with different bankruptcy strictness #
#======================================================#

if Indicator_solve_stationary_equlibria_across_η == true

    η_min_search = 0.20
    η_max_search = 0.30
    η_step_search = 0.05
    var_names, results_A_NFF, results_V_NFF, results_V_pos_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_V_pos_FF, results_μ_FF = results_η_function(η_min = η_min_search, η_max = η_max_search, η_step = η_step_search)
    @save "results_eta.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

end

#=============================#
# Solve transitional dynamics #
#=============================#

if Indicator_solve_transitional_dynamics == true

    @load "results_eta.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

    # specily the new and old policies
    η_old, λ_old = results_A_FF[1,2], results_A_FF[3,2]
    η_new, λ_new = results_A_FF[1,1], results_A_FF[3,1]

    # old stationary equilibrium
    println("Solving old steady state...")
    parameters_old = parameters_function(η = η_old)
    variables_old = variables_function(parameters_old; λ = λ_old)
    solve_economy_function!(variables_old, parameters_old)

    # new stationary equilibrium
    println("Solving new steady state...")
    parameters_new = parameters_function(η = η_new)
    variables_new = variables_function(parameters_new; λ = λ_new)
    solve_economy_function!(variables_new, parameters_new)

    # solve transitional dynamics
    variables_T = variables_T_function(variables_old, variables_new, parameters_new; T_size = 60)
    transitional_dynamic_λ_function!(variables_T, variables_new, parameters_new; iter_max = 10, slow_updating = slow_updating)

end

#============#
# Simulation #
#============#

if Indicator_simulation == true

    # specify parameters
    num_hh = 20000
    num_periods = 2100
    burn_in = 100

    # simulate the model
    variables.q[1:(parameters.a_ind_zero-1),:,:] .= variables.q[1:(parameters.a_ind_zero-1)] .* (1.0 + parameters.τ + variables.aggregate_prices.ι_λ) ./ (1.0 + parameters.τ)
    panel_asset, panel_history, panel_default, panel_age, panel_consumption, shock_ρ, shock_e_1, shock_e_2, shock_e_3, shock_ν = simulation(variables, parameters; num_hh = num_hh, num_periods = num_periods, burn_in = burn_in)

    # save simulation results
    # @save "simulations.jld2" panel_asset panel_history panel_default panel_age panel_consumption shock_ρ shock_e_1 shock_e_2 shock_e_3 shock_ν
    @save "simulations_r_f.jld2" panel_asset panel_history panel_default panel_age panel_consumption shock_ρ shock_e_1 shock_e_2 shock_e_3 shock_ν
    # @save "simulations_NFF.jld2" panel_asset panel_history panel_default panel_age panel_consumption shock_ρ shock_e_1 shock_e_2 shock_e_3 shock_ν

end

if Indicator_simulation_results == true

    # load simulation results
    @load "simulations.jld2" panel_asset panel_history panel_default panel_age panel_consumption shock_ρ shock_e_1 shock_e_2 shock_e_3 shock_ν
    @load "simulations_r_f.jld2" panel_asset panel_history panel_default panel_age panel_consumption shock_ρ shock_e_1 shock_e_2 shock_e_3 shock_ν
    @load "simulations_NFF.jld2" panel_asset panel_history panel_default panel_age panel_consumption shock_ρ shock_e_1 shock_e_2 shock_e_3 shock_ν

    # number of total periods
    period_all = size(panel_asset)[2]

    # share of defaulters
    fraction_default_sim_all_periods = zeros(period_all)
    for i in 1:period_all
        fraction_default_sim_all_periods[i] = sum(panel_default[:,i]) / num_hh * 100
    end
    fraction_default_sim_all_periods_ave = sum(fraction_default_sim_all_periods) / period_all

    # share in debts
    fraction_debts_sim_all_periods = zeros(period_all)
    for i in 1:period_all
        fraction_debts_sim_all_periods[i] = sum(panel_asset[:,i] .< parameters.a_ind_zero) / num_hh * 100
    end
    fraction_debts_sim_all_periods_ave = sum(fraction_debts_sim_all_periods) / period_all

    # consumption
    age_max = 50
    mean_consumption_age = zeros(age_max)
    variance_consumption_age = zeros(age_max)
    panel_log_consumption = log.(panel_consumption)
    mean_log_consumption_age = zeros(age_max)
    variance_log_consumption_age = zeros(age_max)
    for age_i in 1:age_max
        age_bool = (panel_age .== age_i)
        mean_consumption_age[age_i] = sum(panel_consumption[age_bool]) / sum(age_bool)
        variance_consumption_age[age_i] = sum((panel_consumption[age_bool] .- mean_consumption_age[age_i]).^2) / sum(age_bool)
        mean_log_consumption_age[age_i] = sum(panel_log_consumption[age_bool]) / sum(age_bool)
        variance_log_consumption_age[age_i] = sum((panel_log_consumption[age_bool] .- mean_log_consumption_age[age_i]).^2) / sum(age_bool)
    end

    age_max = 80
    mean_consumption_age_r_f = zeros(age_max)
    variance_consumption_age_r_f = zeros(age_max)
    panel_log_consumption_r_f = log.(panel_consumption)
    mean_log_consumption_age_r_f = zeros(age_max)
    variance_log_consumption_age_r_f = zeros(age_max)
    for age_i in 1:age_max
        age_bool_r_f = (panel_age .== age_i)
        mean_consumption_age_r_f[age_i] = sum(panel_consumption[age_bool_r_f]) / sum(age_bool_r_f)
        variance_consumption_age_r_f[age_i] = sum((panel_consumption[age_bool_r_f] .- mean_consumption_age_r_f[age_i]).^2) / sum(age_bool_r_f)
        mean_log_consumption_age_r_f[age_i] = sum(panel_log_consumption_r_f[age_bool_r_f]) / sum(age_bool_r_f)
        variance_log_consumption_age_r_f[age_i] = sum((panel_log_consumption_r_f[age_bool_r_f] .- mean_log_consumption_age_r_f[age_i]).^2) / sum(age_bool_r_f)
    end

    age_max = 50
    mean_consumption_age_NFF = zeros(age_max)
    variance_consumption_age_NFF = zeros(age_max)
    panel_log_consumption_NFF = log.(panel_consumption)
    mean_log_consumption_age_NFF = zeros(age_max)
    variance_log_consumption_age_NFF = zeros(age_max)
    for age_i in 1:age_max
        age_bool_NFF = (panel_age .== age_i)
        mean_consumption_age_NFF[age_i] = sum(panel_consumption[age_bool_NFF]) / sum(age_bool_NFF)
        variance_consumption_age_NFF[age_i] = sum((panel_consumption[age_bool_NFF] .- mean_consumption_age_NFF[age_i]).^2) / sum(age_bool_NFF)
        mean_log_consumption_age_NFF[age_i] = sum(panel_log_consumption_NFF[age_bool_NFF]) / sum(age_bool_NFF)
        variance_log_consumption_age_NFF[age_i] = sum((panel_log_consumption_NFF[age_bool_NFF] .- mean_log_consumption_age_NFF[age_i]).^2) / sum(age_bool_NFF)
    end

    plot_consumption = plot(1:age_max, mean_consumption_age, legend=:bottomright, label="with financial frictions", xlabel="working age", ylabel="consumption")
    # plot!(1:age_max, mean_consumption_age_r_f, label="with financial frictions (fixed r_f)")
    plot_consumption = plot!(1:age_max, mean_consumption_age_NFF, label="without financial frictions")
    Plots.savefig(plot_consumption, pwd() * "\\figures\\plot_consumption.pdf")

    age_max_ind = 5
    mean_consumption_age_adj = mean_consumption_age .- mean(mean_consumption_age)
    mean_consumption_age_NFF_adj = mean_consumption_age_NFF .- mean(mean_consumption_age_NFF)
    plot(1:age_max_ind, mean_consumption_age_adj[1:age_max_ind], legend=:bottomright, label="with financial frictions", xlabel="age", ylabel="consumption")
    # plot!(1:age_max, mean_consumption_age_r_f, label="with financial frictions (fixed r_f)")
    plot!(1:age_max_ind, mean_consumption_age_NFF_adj[1:age_max_ind], label="without financial frictions")

    plot_var_log_consumption = plot(1:age_max, variance_log_consumption_age, legend=:bottomright, label="with financial frictions", xlabel="working age", ylabel="variance of log consumption")
    # plot!(1:age_max, variance_log_consumption_age_r_f, label="with financial frictions (fixed r_f)")
    plot_var_log_consumption = plot!(1:age_max, variance_log_consumption_age_NFF, label="without financial frictions")
    Plots.savefig(plot_var_log_consumption, pwd() * "\\figures\\plot_var_log_consumption.pdf")

end
