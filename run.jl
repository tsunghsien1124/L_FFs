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
Indicator_solve_stationary_equlibria_across_p_h = false
Indicator_solve_transitional_dynamics_across_η = false
Indicator_solve_transitional_dynamics_across_p_h = false
Indicator_simulation_benchmark = true
Indicator_simulation_benchmark_results = false

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
# plot(parameters_new.a_grid_neg[a_neg_index:end], variables_new.q[a_neg_index:parameters_new.a_ind_zero,2,:], legend=:none)

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

if Indicator_solve_stationary_equlibria_across_p_h == true

    p_h_min_search = 8.0
    p_h_max_search = 12.0
    p_h_step_search = 2.0
    var_names, results_A_NFF, results_V_NFF, results_V_pos_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_V_pos_FF, results_μ_FF = results_p_h_function(p_h_min = p_h_min_search, p_h_max = p_h_max_search, p_h_step = p_h_step_search)
    @save "results_p_h.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

end

#=============================#
# Solve transitional dynamics #
#=============================#

if Indicator_solve_transitional_dynamics_across_η == true

    # load stationary equilibria across η
    @load "results_eta.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

    # specily the new and old policies
    η_20, λ_20 = results_A_FF[1,3], results_A_FF[3,3] # η = 0.20
    η_25, λ_25 = results_A_FF[1,2], results_A_FF[3,2] # η = 0.25
    η_30, λ_30 = results_A_FF[1,1], results_A_FF[3,1] # η = 0.30

    # stationary equilibrium when η = 0.20
    println("Solving steady state when η = $η_20...")
    parameters_20 = parameters_function(η = η_20)
    variables_20 = variables_function(parameters_20; λ = λ_20)
    solve_economy_function!(variables_20, parameters_20)

    # stationary equilibrium when η = 0.25
    println("Solving steady state when η = $η_25...")
    parameters_25 = parameters_function(η = η_25)
    variables_25 = variables_function(parameters_25; λ = λ_25)
    solve_economy_function!(variables_25, parameters_25)

    # stationary equilibrium when η = 0.30
    println("Solving steady state when η = $η_30...")
    parameters_30 = parameters_function(η = η_30)
    variables_30 = variables_function(parameters_30; λ = λ_30)
    solve_economy_function!(variables_30, parameters_30)

    # set parameters for computation
    load_initial_value = true
    if load_initial_value == true
        @load "results_transition_eta_1E-3.jld2" transtion_path_eta_25_30 transtion_path_eta_25_20
    end
    T_size = 200
    T_degree = 7.0
    iter_max = 1000
    tol = 1E-3
    slow_updating_transitional_dynamics = 0.05

    # from η = 0.25 to η = 0.30
    println("Solving transitions from η = $η_25 to η = $η_30...")
    if load_initial_value == true
        variables_T_25_30 = variables_T_function(transtion_path_eta_25_30, variables_25, variables_30, parameters_30)
    else
        variables_T_25_30 = variables_T_function(variables_25, variables_30, parameters_30; T_size = T_size, T_degree = T_degree)
    end
    transitional_dynamic_λ_function!(variables_T_25_30, variables_25, variables_30, parameters_30; tol = tol, iter_max = iter_max, slow_updating = slow_updating_transitional_dynamics)
    transtion_path_eta_25_30 = variables_T_25_30.aggregate_prices.leverage_ratio_λ

    # from η = 0.25 to η = 0.20
    println("Solving transitions from η = $η_25 to η = $η_20...")
    if load_initial_value == true
        variables_T_25_20 = variables_T_function(transtion_path_eta_25_20, variables_25, variables_20, parameters_30)
    else
        variables_T_25_20 = variables_T_function(variables_25, variables_20, parameters_20; T_size = T_size, T_degree = T_degree)
    end
    transitional_dynamic_λ_function!(variables_T_25_20, variables_25, variables_20, parameters_20; tol = tol, iter_max = iter_max, slow_updating = slow_updating_transitional_dynamics)
    transtion_path_eta_25_20 = variables_T_25_20.aggregate_prices.leverage_ratio_λ

    # save transition path
    @save "results_transition_eta.jld2" transtion_path_eta_25_30 transtion_path_eta_25_20

end

if Indicator_solve_transitional_dynamics_across_p_h == true

    @load "results_p_h.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

    # specily the new and old policies
    p_h_8, λ_8 = results_A_FF[1,3], results_A_FF[3,3] # p_h = 1 / 8
    p_h_10, λ_10 = results_A_FF[1,2], results_A_FF[3,2] # p_h = 1 / 10
    p_h_12, λ_12 = results_A_FF[1,1], results_A_FF[3,1] # p_h = 1 / 12

    # stationary equilibrium when p_h = 1 / 8
    println("Solving steady state when p_h = $p_h_8...")
    parameters_8 = parameters_function(p_h = p_h_8)
    variables_8 = variables_function(parameters_8; λ = λ_8)
    solve_economy_function!(variables_8, parameters_8)

    # stationary equilibrium when p_h = 1 / 10
    println("Solving steady state when p_h = $p_h_10...")
    parameters_10 = parameters_function(p_h = p_h_10)
    variables_10 = variables_function(parameters_10; λ = λ_10)
    solve_economy_function!(variables_10, parameters_10)

    # stationary equilibrium when p_h = 1 / 12
    println("Solving steady state when p_h = $p_h_12...")
    parameters_12 = parameters_function(p_h = p_h_12)
    variables_12 = variables_function(parameters_12; λ = λ_12)
    solve_economy_function!(variables_12, parameters_12)

    # set parameters for computation
    load_initial_value = false
    if load_initial_value == true
        @load "results_transition_p_h.jld2" transtion_path_p_h_10_12 transtion_path_p_h_10_8
    end
    T_size = 200
    T_degree = 7.0
    iter_max = 1000
    tol = 1E-3
    slow_updating_transitional_dynamics = 0.05

    # from p_h = 1 / 10 to p_h = 1 / 12
    println("Solving transitions from p_h = $p_h_10 to p_h = $p_h_12...")
    if load_initial_value == true
        variables_T_10_12 = variables_T_function(transtion_path_p_h_10_12, variables_10, variables_12, parameters_12)
    else
        variables_T_10_12 = variables_T_function(variables_10, variables_12, parameters_12; T_size = T_size, T_degree = T_degree)
    end
    transitional_dynamic_λ_function!(variables_T_10_12, variables_10, variables_12, parameters_12; tol = tol, iter_max = iter_max, slow_updating = slow_updating_transitional_dynamics)
    transtion_path_p_h_10_12 = variables_T_10_12.aggregate_prices.leverage_ratio_λ

    # from p_h = 1 / 10 to p_h = 1 / 8
    println("Solving transitions from p_h = $p_h_10 to p_h = $p_h_8...")
    if load_initial_value == true
        variables_T_10_8 = variables_T_function(transtion_path_p_h_10_8, variables_10, variables_8, parameters_8)
    else
        variables_T_10_8 = variables_T_function(variables_10, variables_8, parameters_8; T_size = T_size, T_degree = T_degree)
    end
    transitional_dynamic_λ_function!(variables_T_10_8, variables_10, variables_8, parameters_8; tol = tol, iter_max = iter_max, slow_updating = slow_updating_transitional_dynamics)
    transtion_path_p_h_10_8 = variables_T_10_8.aggregate_prices.leverage_ratio_λ

    # save transition path
    @save "results_transition_p_h.jld2" transtion_path_p_h_10_12 transtion_path_p_h_10_8

end

#============#
# Simulation #
#============#

if Indicator_simulation_benchmark == true

    # specify parameters
    num_hh = 40000
    num_periods = 5100
    burn_in = 100

    # load stationary equilibria across η
    @load "results_eta.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

    # extract equilibria with financial fricitons
    η, λ_FF = results_A_FF[1,2], results_A_FF[3,2]

    # stationary equilibrium when η = 0.25 with and without financial frictions
    parameters = parameters_function(η = η)
    println("Solving steady state with benchmark calibration (η = $η) and financial frictions...")
    variables_FF = variables_function(parameters; λ = λ_FF)
    solve_economy_function!(variables_FF, parameters)
    println("Solving steady state with benchmark calibration (η = $η) and no financial frictions...")
    variables_NFF = variables_function(parameters; λ = 0.0)
    solve_economy_function!(variables_NFF, parameters)

    # simulate models
    panel_asset_FF, panel_history_FF, panel_default_FF, panel_age_FF, panel_consumption_FF, shock_ρ_FF, shock_e_1_FF, shock_e_2_FF, shock_e_3_FF, shock_ν_FF = simulation(variables_FF, parameters; num_hh = num_hh, num_periods = num_periods, burn_in = burn_in)
    panel_asset_NFF, panel_history_NFF, panel_default_NFF, panel_age_NFF, panel_consumption_NFF, shock_ρ_NFF, shock_e_1_NFF, shock_e_2_NFF, shock_e_3_NFF, shock_ν_NFF = simulation(variables_NFF, parameters; num_hh = num_hh, num_periods = num_periods, burn_in = burn_in)

    # save simulation results
    @save "simulations_benchmark_FF.jld2" panel_asset_FF panel_history_FF panel_default_FF panel_age_FF panel_consumption_FF shock_ρ_FF shock_e_1_FF shock_e_2_FF shock_e_3_FF shock_ν_FF
    @save "simulations_benchmark_NFF.jld2" panel_asset_NFF panel_history_NFF panel_default_NFF panel_age_NFF panel_consumption_NFF shock_ρ_NFF shock_e_1_NFF shock_e_2_NFF shock_e_3_NFF shock_ν_NFF

end

if Indicator_simulation_benchmark_results == true

    # load simulation results
    @load "simulations_benchmark_FF.jld2" panel_asset_FF panel_history_FF panel_default_FF panel_age_FF panel_consumption_FF shock_ρ_FF shock_e_1_FF shock_e_2_FF shock_e_3_FF shock_ν_FF
    @load "simulations_benchmark_NFF.jld2" panel_asset_NFF panel_history_NFF panel_default_NFF panel_age_NFF panel_consumption_NFF shock_ρ_NFF shock_e_1_NFF shock_e_2_NFF shock_e_3_NFF shock_ν_NFF

    # set parameters
    parameters = parameters_function()
    num_periods = size(panel_asset_FF)[2]
    num_hh = size(panel_asset_FF)[1]

    # share of defaulters
    fraction_default_sim_FF = zeros(num_periods)
    fraction_default_sim_NFF = zeros(num_periods)
    for i in 1:num_periods
        fraction_default_sim_FF[i] = sum(panel_default_FF[:,i]) / num_hh * 100
        fraction_default_sim_NFF[i] = sum(panel_default_NFF[:,i]) / num_hh * 100
    end
    fraction_default_sim_FF_avg = sum(fraction_default_sim_FF) / num_periods
    fraction_default_sim_NFF_avg = sum(fraction_default_sim_NFF) / num_periods

    # share in debts
    fraction_debts_sim_FF = zeros(num_periods)
    fraction_debts_sim_NFF = zeros(num_periods)
    for i in 1:num_periods
        fraction_debts_sim_FF[i] = sum(panel_asset_FF[:,i] .< parameters.a_ind_zero) / num_hh * 100
        fraction_debts_sim_NFF[i] = sum(panel_asset_NFF[:,i] .< parameters.a_ind_zero) / num_hh * 100
    end
    fraction_debts_sim_FF_avg = sum(fraction_debts_sim_FF) / num_periods
    fraction_debts_sim_NFF_avg = sum(fraction_debts_sim_NFF) / num_periods

    # share of defaulters, conditional borrowing
    fraction_cond_default_sim_FF_avg = fraction_default_sim_FF_avg / fraction_debts_sim_FF_avg * 100
    fraction_cond_default_sim_NFF_avg = fraction_default_sim_NFF_avg / fraction_debts_sim_NFF_avg * 100

    # consumption over life cycle
    age_max = 50
    mean_consumption_age_FF, mean_consumption_age_NFF = zeros(age_max), zeros(age_max)
    variance_consumption_age_FF, variance_consumption_age_NFF = zeros(age_max), zeros(age_max)
    panel_log_consumption_FF, panel_log_consumption_NFF = log.(panel_consumption_FF), log.(panel_consumption_NFF)
    mean_log_consumption_age_FF, mean_log_consumption_age_NFF = zeros(age_max), zeros(age_max)
    variance_log_consumption_age_FF, variance_log_consumption_age_NFF = zeros(age_max), zeros(age_max)
    for age_i in 1:age_max
        age_bool_FF = (panel_age_FF .== age_i)
        mean_consumption_age_FF[age_i] = sum(panel_consumption_FF[age_bool_FF]) / sum(age_bool_FF)
        variance_consumption_age_FF[age_i] = sum((panel_consumption_FF[age_bool_FF] .- mean_consumption_age_FF[age_i]).^2) / sum(age_bool_FF)
        mean_log_consumption_age_FF[age_i] = sum(panel_log_consumption_FF[age_bool_FF]) / sum(age_bool_FF)
        variance_log_consumption_age_FF[age_i] = sum((panel_log_consumption_FF[age_bool_FF] .- mean_log_consumption_age_FF[age_i]).^2) / sum(age_bool_FF)
        age_bool_NFF = (panel_age_NFF .== age_i)
        mean_consumption_age_NFF[age_i] = sum(panel_consumption_NFF[age_bool_NFF]) / sum(age_bool_NFF)
        variance_consumption_age_NFF[age_i] = sum((panel_consumption_NFF[age_bool_NFF] .- mean_consumption_age_NFF[age_i]).^2) / sum(age_bool_NFF)
        mean_log_consumption_age_NFF[age_i] = sum(panel_log_consumption_NFF[age_bool_NFF]) / sum(age_bool_NFF)
        variance_log_consumption_age_NFF[age_i] = sum((panel_log_consumption_NFF[age_bool_NFF] .- mean_log_consumption_age_NFF[age_i]).^2) / sum(age_bool_NFF)
    end

    plot_consumption = plot(1:age_max, mean_consumption_age_FF, legend=:bottomright, label="with financial frictions", xlabel="working age", ylabel="consumption")
    plot_consumption = plot!(1:age_max, mean_consumption_age_NFF, label="without financial frictions")
    plot_consumption
    Plots.savefig(plot_consumption, pwd() * "\\figures\\plot_consumption.pdf")

    plot_var_log_consumption = plot(1:age_max, variance_log_consumption_age_FF, legend=:bottomright, label="with financial frictions", xlabel="working age", ylabel="variance of log consumption")
    plot_var_log_consumption = plot!(1:age_max, variance_log_consumption_age_NFF, label="without financial frictions")
    plot_var_log_consumption
    Plots.savefig(plot_var_log_consumption, pwd() * "\\figures\\plot_var_log_consumption.pdf")

end
