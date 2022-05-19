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

Indicator_decomposition = false

Indicator_solve_stationary_equlibria_across_η = false
Indicator_solve_stationary_equlibria_across_p_h = false
Indicator_solve_stationary_equlibria_across_θ = false
Indicator_solve_stationary_equlibria_across_ψ = false

Indicator_solve_transitional_dynamics_across_η = false
Indicator_solve_transitional_dynamics_across_η_general = false
Indicator_solve_transitional_dynamics_across_p_h = false

Indicator_solve_transitional_dynamics_MIT_z = false

Indicator_simulation_benchmark = false
Indicator_simulation_benchmark_results = false
Indicator_simulation_across_θ = false
Indicator_simulation_across_θ_results = false
Indicator_simulation_across_ψ = false
Indicator_simulation_across_ψ_results = false

# print out the number of threads
println("Julia is running with $(Threads.nthreads()) threads...")
slow_updating = 1.0

#==============================#
# Solve stationary equilibrium #
#==============================#

if Indicator_solve_equlibria_λ_min_and_max == true

    parameters = parameters_function()

    variables_min = variables_function(parameters; λ = 0.0)
    ED_KL_to_D_ratio_min, ED_leverage_ratio_min, crit_V_min, crit_μ_min = solve_economy_function!(variables_min, parameters; slow_updating = slow_updating)

    variables_max = variables_function(parameters; λ = 1.0 - sqrt(parameters.ψ))
    ED_KL_to_D_ratio_max, ED_leverage_ratio_max, crit_V_max, crit_μ_max = solve_economy_function!(variables_max, parameters; slow_updating = slow_updating)

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
        "convergence of V" crit_V_min crit_V_max
        "convergence of μ" crit_μ_min crit_μ_max
    ]
    pretty_table(data_spec; header = ["Name", "λ minimum", "λ maximum"], alignment = [:c, :c, :c], formatters = ft_round(8), body_hlines = [3,5,9,10])

end

if Indicator_solve_equlibrium_given_λ == true

    parameters = parameters_function()
    variables = variables_function(parameters; λ = 0.0267604155273437)
    ED_KL_to_D_ratio, ED_leverage_ratio, crit_V, crit_μ = solve_economy_function!(variables, parameters; slow_updating = slow_updating)
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
        flag,
        crit_V,
        crit_μ
        ]
    display(calibration_results)

end

#================#
# Checking Plots #
#================#
# a_neg_index = 1
# plot(parameters.a_grid_neg[a_neg_index:end], variables.q[a_neg_index:parameters.a_ind_zero,2,:], legend=:none)
# plot(parameters.a_grid_neg[a_neg_index:end], variables.policy_d[a_neg_index:parameters.a_ind_zero,1,:,1,2], legend=:none)

#============================================#
# Solve stationary equilibrium (calibration) #
#============================================#
if Indicator_solve_stationary_equlibrium == true

    β_search = 0.940 / 0.980 # collect(0.94:0.01:0.97)
    ζ_d_search = 0.0215 # collect(0.03000:0.00500:0.0350)
    ν_p_search = 0.01057 # collect(0.010400:0.000100:0.010500)

    β_search_size = length(β_search)
    ζ_d_search_size = length(ζ_d_search)
    ν_p_search_size = length(ν_p_search)
    search_size = β_search_size * ζ_d_search_size * ν_p_search_size
    calibration_results = zeros(search_size, 21)

    for β_i in 1:β_search_size, ζ_d_i in 1:ζ_d_search_size, ν_p_i in 1:ν_p_search_size

        parameters = parameters_function(β = β_search[β_i], ζ_d = ζ_d_search[ζ_d_i], ν_p = ν_p_search[ν_p_i])
        variables_λ_lower, variables, flag, crit_V, crit_μ = optimal_multiplier_function(parameters; slow_updating = slow_updating)

        search_iter = (β_i - 1)*(ζ_d_search_size*ν_p_search_size) + (ζ_d_i-1)*ν_p_search_size + ν_p_i

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
        calibration_results[search_iter, 20] = crit_V
        calibration_results[search_iter, 21] = crit_μ

    end

    CSV.write("calibration_julia.csv", Tables.table(calibration_results), header=false)

end

#===============#
# Decomposition #
#===============#

if Indicator_decomposition == true

    # set up parameters
    parameters = parameters_function()

    # benchmark (with financial frictions) --- incentive and divestment channels
    variables_benchmark = variables_function(parameters; λ = 0.0267604155273437)
    solve_economy_function!(variables_benchmark, parameters; slow_updating = slow_updating)

    # without financial frictions --- no incentive and divestment channels
    variables_NFF = variables_function(parameters; λ = 0.0)
    solve_economy_function!(variables_NFF, parameters; slow_updating = slow_updating)

    # with extra fixed costs --- only incentive channel
    parameters_τ = parameters_function(τ = parameters.r_f + variables_benchmark.aggregate_prices.ι_λ)
    variables_τ = variables_function(parameters_τ; λ = 0.0)
    solve_economy_function!(variables_τ, parameters_τ; slow_updating = slow_updating)

    # collect results
    results = Any[
        "" "Without Financial Frictions" "Without Financial Frictions / Higher Lending Costs" "With Financial Frictions"
        "" "" "" ""
        "Incentive premium (%)" variables_NFF.aggregate_prices.ι_λ*100 variables_τ.aggregate_prices.ι_λ*100 variables_benchmark.aggregate_prices.ι_λ*100
        "Lending costs (%)" (parameters.r_f + variables_NFF.aggregate_prices.ι_λ)*100 parameters_τ.τ*100 (parameters.r_f + variables_benchmark.aggregate_prices.ι_λ)*100
        "" "" "" ""
        "Wage" variables_NFF.aggregate_prices.w_λ variables_τ.aggregate_prices.w_λ variables_benchmark.aggregate_prices.w_λ
        "" "" "" ""
        "Share in debts (%)" variables_NFF.aggregate_variables.share_in_debts*100 variables_τ.aggregate_variables.share_in_debts*100 variables_benchmark.aggregate_variables.share_in_debts*100
        "Conditional default rate (%)"  (variables_NFF.aggregate_variables.share_of_filers/variables_NFF.aggregate_variables.share_in_debts)*100 (variables_τ.aggregate_variables.share_of_filers/variables_τ.aggregate_variables.share_in_debts)*100 (variables_benchmark.aggregate_variables.share_of_filers/variables_benchmark.aggregate_variables.share_in_debts)*100
        "Debt-to-earnings ratio (%)" variables_NFF.aggregate_variables.debt_to_earning_ratio*100 variables_τ.aggregate_variables.debt_to_earning_ratio*100 variables_benchmark.aggregate_variables.debt_to_earning_ratio*100
        "" "" "" ""
        "Average interest rate (%)" variables_NFF.aggregate_variables.avg_loan_rate*100 variables_τ.aggregate_variables.avg_loan_rate*100 variables_benchmark.aggregate_variables.avg_loan_rate*100
    ]

    # save results
    CSV.write("decomposition.csv", Tables.table(results), header = false)

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

    η_min_search = 0.30
    η_max_search = 0.80
    η_step_search = 0.10
    var_names, results_A_NFF, results_V_NFF, results_V_pos_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_V_pos_FF, results_μ_FF = results_η_function(η_min = η_min_search, η_max = η_max_search, η_step = η_step_search)
    @save "results_eta_0.3_0.8.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

    η_min_search = 0.30
    η_max_search = 0.40
    η_step_search = 0.02
    var_names, results_A_NFF, results_V_NFF, results_V_pos_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_V_pos_FF, results_μ_FF = results_η_function(η_min = η_min_search, η_max = η_max_search, η_step = η_step_search)
    @save "results_eta_0.3_0.4_step_0.02.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

end

if Indicator_solve_stationary_equlibria_across_p_h == true

    p_h_min_search = 5.0
    p_h_max_search = 15.0
    p_h_step_search = 5.0
    var_names, results_A_NFF, results_V_NFF, results_V_pos_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_V_pos_FF, results_μ_FF = results_p_h_function(p_h_min = p_h_min_search, p_h_max = p_h_max_search, p_h_step = p_h_step_search)
    @save "results_p_h.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

end

if Indicator_solve_stationary_equlibria_across_θ == true

    θ_min_search = 1.0/3.0 * 0.9
    θ_max_search = 1.0/3.0 * 1.1
    θ_step_search = 1.0/3.0 * 0.1
    var_names, results_A_NFF, results_V_NFF, results_V_pos_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_V_pos_FF, results_μ_FF = results_θ_function(θ_min = θ_min_search, θ_max = θ_max_search, θ_step = θ_step_search)
    @save "results_theta.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

end

if Indicator_solve_stationary_equlibria_across_ψ == true

    ψ_min_search = 18
    ψ_max_search = 22
    ψ_step_search = 2
    var_names, results_A_NFF, results_V_NFF, results_V_pos_NFF, results_μ_NFF, results_A_FF, results_V_FF, results_V_pos_FF, results_μ_FF = results_ψ_function(ψ_min = ψ_min_search, ψ_max = ψ_max_search, ψ_step = ψ_step_search)
    @save "results_psi.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

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

    # printout results of aggregate statistics
    results_equilibria_across_η = Any[
        "" "eta = 0.20" "eta = 0.25" "eta = 0.30"
        "" "" "" ""
        "Banking leverage ratio" variables_20.aggregate_variables.leverage_ratio variables_25.aggregate_variables.leverage_ratio variables_30.aggregate_variables.leverage_ratio
        "Leverage premium (%)" variables_20.aggregate_prices.ι_λ*100 variables_25.aggregate_prices.ι_λ*100 variables_30.aggregate_prices.ι_λ*100
        "Wage" variables_20.aggregate_prices.w_λ variables_25.aggregate_prices.w_λ variables_30.aggregate_prices.w_λ
        "" "" "" ""
        "Default rate (%)" variables_20.aggregate_variables.share_of_filers*100 variables_25.aggregate_variables.share_of_filers*100 variables_30.aggregate_variables.share_of_filers*100
        "Share in debt (%)" variables_20.aggregate_variables.share_in_debts*100 variables_25.aggregate_variables.share_in_debts*100 variables_30.aggregate_variables.share_in_debts*100
        "Debt-to-earnings ratio (%)" variables_20.aggregate_variables.debt_to_earning_ratio*100 variables_25.aggregate_variables.debt_to_earning_ratio*100 variables_30.aggregate_variables.debt_to_earning_ratio*100
        "Average interest rate (%)" variables_20.aggregate_variables.avg_loan_rate*100 variables_25.aggregate_variables.avg_loan_rate*100 variables_30.aggregate_variables.avg_loan_rate*100
    ]
    display(results_equilibria_across_η)

    # save results
    CSV.write("results_equilibria_across_eta.csv", Tables.table(results_equilibria_across_η), header = false)

    # set parameters for computation
    T_size = 80
    T_degree = 15.0
    iter_max = 500
    tol = 1E-8
    slow_updating_transitional_dynamics = 0.1
    load_initial_value = true
    if load_initial_value == true
        @load "results_transition_eta.jld2" transtion_path_eta_25_30 transtion_path_eta_25_20
    end
    initial_z = ones(T_size+2)

    # from η = 0.25 to η = 0.30
    println("Solving transitions from η = $η_25 to η = $η_30...")
    if load_initial_value == true
        variables_T_25_30 = variables_T_function(transtion_path_eta_25_30, initial_z, variables_25, variables_30, parameters_30)
    else
        variables_T_25_30 = variables_T_function(variables_25, variables_30, parameters_30; T_size = T_size, T_degree = T_degree)
    end
    transitional_dynamic_λ_function!(variables_T_25_30, variables_25, variables_30, parameters_30; tol = tol, iter_max = iter_max, slow_updating = slow_updating_transitional_dynamics)
    transtion_path_eta_25_30 = variables_T_25_30.aggregate_prices.leverage_ratio_λ
    plot_transtion_path_eta_25_30 = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "", xlabel = "Time")
    plot_transtion_path_eta_25_30 = plot!(transtion_path_eta_25_30, linecolor = :blue, linewidth = 3, legend=:none)
    plot_transtion_path_eta_25_30
    Plots.savefig(plot_transtion_path_eta_25_30, pwd() * "\\figures\\plot_transtion_path_eta_25_30.pdf")

    # from η = 0.25 to η = 0.20
    println("Solving transitions from η = $η_25 to η = $η_20...")
    if load_initial_value == true
        variables_T_25_20 = variables_T_function(transtion_path_eta_25_20, initial_z, variables_25, variables_20, parameters_30)
    else
        variables_T_25_20 = variables_T_function(variables_25, variables_20, parameters_20; T_size = T_size, T_degree = T_degree)
    end
    transitional_dynamic_λ_function!(variables_T_25_20, variables_25, variables_20, parameters_20; tol = tol, iter_max = iter_max, slow_updating = slow_updating_transitional_dynamics)
    transtion_path_eta_25_20 = variables_T_25_20.aggregate_prices.leverage_ratio_λ
    plot_transtion_path_eta_25_20 = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "", xlabel = "Time")
    plot_transtion_path_eta_25_20 = plot!(transtion_path_eta_25_20, linecolor = :blue, linewidth = 3, legend=:none)
    Plots.savefig(plot_transtion_path_eta_25_20, pwd() * "\\figures\\plot_transtion_path_eta_25_20.pdf")

    # save transition path
    @save "results_transition_eta.jld2" transtion_path_eta_25_30 transtion_path_eta_25_20

    # compute welfare metrics from η = 0.25 to η = 0.30
    welfare_CEV_25_30_good_with_debt = 100 * sum(((variables_T_25_30.V[1:(parameters_25.a_ind_zero-1),:,:,:,:,2] ./ variables_25.V[1:(parameters_25.a_ind_zero-1),:,:,:,:]) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* (variables_25.μ[1:(parameters_25.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_25.μ[1:(parameters_25.a_ind_zero-1),:,:,:,:,1])))
    welfare_CEV_25_30_good_no_debt = 100 * sum(((variables_T_25_30.V[parameters_25.a_ind_zero:end,:,:,:,:,2] ./ variables_25.V[parameters_25.a_ind_zero:end,:,:,:,:]) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* (variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,1])))
    welfare_CEV_25_30_good = 100 * sum(((variables_T_25_30.V[:,:,:,:,:,2] ./ variables_25.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* (variables_25.μ[:,:,:,:,:,1] ./ sum(variables_25.μ[:,:,:,:,:,1])))
    welfare_CEV_25_30_bad =  100 * sum(((variables_T_25_30.V_pos[:,:,:,:,:,2] ./ variables_25.V_pos) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* (variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2])))
    welfare_CEV_25_30 = 100 * (sum(((variables_T_25_30.V[:,:,:,:,:,2] ./ variables_25.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* variables_25.μ[:,:,:,:,:,1]) + sum(((variables_T_25_30.V_pos[:,:,:,:,:,2] ./ variables_25.V_pos) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2]))

    welfare_favor_25_30_good_with_debt = 100 * sum((variables_T_25_30.V[1:(parameters_25.a_ind_zero-1),:,:,:,:,2] .>= variables_25.V[1:(parameters_25.a_ind_zero-1),:,:,:,:]) .* (variables_25.μ[1:(parameters_25.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_25.μ[1:(parameters_25.a_ind_zero-1),:,:,:,:,1])))
    welfare_favor_25_30_good_without_debt = 100 * sum((variables_T_25_30.V[parameters_25.a_ind_zero:end,:,:,:,:,2] .>= variables_25.V[parameters_25.a_ind_zero:end,:,:,:,:]) .* (variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,1])))
    welfare_favor_25_30_good = 100 * sum((variables_T_25_30.V[:,:,:,:,:,2] .>= variables_25.V) .* (variables_25.μ[:,:,:,:,:,1] ./ sum(variables_25.μ[:,:,:,:,:,1])))
    welfare_favor_25_30_bad = 100 * sum((variables_T_25_30.V_pos[:,:,:,:,:,2] .>= variables_25.V_pos) .* (variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2])))
    welfare_favor_25_30 = 100 * (sum((variables_T_25_30.V[:,:,:,:,:,2] .>= variables_25.V) .* variables_25.μ[:,:,:,:,:,1]) + sum((variables_T_25_30.V_pos[:,:,:,:,:,2] .>= variables_25.V_pos) .* variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2]))

    # compute welfare metrics from η = 0.25 to η = 0.20
    welfare_CEV_25_20_good_with_debt = 100 * sum(((variables_T_25_20.V[1:(parameters_25.a_ind_zero-1),:,:,:,:,2] ./ variables_25.V[1:(parameters_25.a_ind_zero-1),:,:,:,:]) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* (variables_25.μ[1:(parameters_25.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_25.μ[1:(parameters_25.a_ind_zero-1),:,:,:,:,1])))
    welfare_CEV_25_20_good_no_debt = 100 * sum(((variables_T_25_20.V[parameters_25.a_ind_zero:end,:,:,:,:,2] ./ variables_25.V[parameters_25.a_ind_zero:end,:,:,:,:]) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* (variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,1])))
    welfare_CEV_25_20_good = 100 * sum(((variables_T_25_20.V[:,:,:,:,:,2] ./ variables_25.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* (variables_25.μ[:,:,:,:,:,1] ./ sum(variables_25.μ[:,:,:,:,:,1])))
    welfare_CEV_25_20_bad =  100 * sum(((variables_T_25_20.V_pos[:,:,:,:,:,2] ./ variables_25.V_pos) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* (variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2])))
    welfare_CEV_25_20 = 100 * (sum(((variables_T_25_20.V[:,:,:,:,:,2] ./ variables_25.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* variables_25.μ[:,:,:,:,:,1]) + sum(((variables_T_25_20.V_pos[:,:,:,:,:,2] ./ variables_25.V_pos) .^ (1.0/(1.0-parameters_25.σ)) .- 1.0) .* variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2]))

    welfare_favor_25_20_good_with_debt = 100 * sum((variables_T_25_20.V[1:(parameters_25.a_ind_zero-1),:,:,:,:,2] .>= variables_25.V[1:(parameters_25.a_ind_zero-1),:,:,:,:]) .* (variables_25.μ[1:(parameters_25.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_25.μ[1:(parameters_25.a_ind_zero-1),:,:,:,:,1])))
    welfare_favor_25_20_good_without_debt = 100 * sum((variables_T_25_20.V[parameters_25.a_ind_zero:end,:,:,:,:,2] .>= variables_25.V[parameters_25.a_ind_zero:end,:,:,:,:]) .* (variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,1])))
    welfare_favor_25_20_good = 100 * sum((variables_T_25_20.V[:,:,:,:,:,2] .>= variables_25.V) .* (variables_25.μ[:,:,:,:,:,1] ./ sum(variables_25.μ[:,:,:,:,:,1])))
    welfare_favor_25_20_bad = 100* sum((variables_T_25_20.V_pos[:,:,:,:,:,2] .>= variables_25.V_pos) .* (variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2])))
    welfare_favor_25_20 = 100 * (sum((variables_T_25_20.V[:,:,:,:,:,2] .>= variables_25.V) .* variables_25.μ[:,:,:,:,:,1]) + sum((variables_T_25_20.V_pos[:,:,:,:,:,2] .>= variables_25.V_pos) .* variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,2]))

    # share of households
    HHs_good_debt = 100 * sum(variables_25.μ[1:(parameters_25.a_ind_zero-1),:,:,:,:,1])
    HHs_good_no_debt = 100 * sum(variables_25.μ[parameters_25.a_ind_zero:end,:,:,:,:,1])
    HHs_good = HHs_good_debt + HHs_good_no_debt
    HHs_good_debt_cond = HHs_good_debt / HHs_good * 100
    HHs_good_no_debt_cond = HHs_good_no_debt / HHs_good * 100
    HHs_bad = 100 * sum(variables_25.μ[:,:,:,:,:,2])
    HHs_total = HHs_good + HHs_bad

    # printout results of welfare effects
    results_welfare_across_η = Any[
        "" "" "from eta = 0.25 to 0.20" "" "from eta = 0.25 to 0.30" ""
        "(%)" "Proportion" "C.E.V." "Favor Reform" "C.E.V." "Favor Reform"
        "" "" "" "" "" ""
        "Have good credit history" HHs_good welfare_CEV_25_20_good welfare_favor_25_20_good welfare_CEV_25_30_good welfare_favor_25_30_good
        "Indebted" HHs_good_debt_cond welfare_CEV_25_20_good_with_debt welfare_favor_25_20_good_with_debt welfare_CEV_25_30_good_with_debt welfare_favor_25_30_good_with_debt
        "Not indebted" HHs_good_no_debt_cond welfare_CEV_25_20_good_no_debt welfare_favor_25_20_good_without_debt welfare_CEV_25_30_good_no_debt welfare_favor_25_30_good_without_debt
        "" "" "" "" "" ""
        "Have bad credit history" HHs_bad welfare_CEV_25_20_bad welfare_favor_25_20_bad welfare_CEV_25_30_bad welfare_favor_25_30_bad
        "" "" "" "" "" ""
        "Total" HHs_total welfare_CEV_25_20 welfare_favor_25_20 welfare_CEV_25_30 welfare_favor_25_30
    ]
    display(results_welfare_across_η)

    # save results
    CSV.write("results_welfare_across_eta.csv", Tables.table(results_welfare_across_η), header = false)

end

if Indicator_solve_transitional_dynamics_across_η_general == true

    #=========================================#
    # combine all jld2 files with different η #
    #=========================================#
    Indicator_merge_differet_jld2 = false

    if Indicator_merge_differet_jld2 == true

        @load "results_eta.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF
        results_A_NFF_all = similar(results_A_NFF)
        copyto!(results_A_NFF_all, results_A_NFF)
        results_A_FF_all = similar(results_A_FF)
        copyto!(results_A_FF_all, results_A_FF)

        @load "results_eta_0.3_0.8.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF
        η_already = results_A_NFF_all[1,:]
        η_new = results_A_NFF[1,:]
        η_new_index = []
        for η_new_i in 1:length(η_new)
            if all(η_already .!= η_new[η_new_i])
                append!(η_new_index, [η_new_i])
            end
        end
        results_A_NFF_all = hcat(results_A_NFF_all, results_A_NFF[:,η_new_index])
        results_A_NFF_all = results_A_NFF_all[:, sortperm(results_A_NFF_all[1,:], rev = true)]
        results_A_FF_all = hcat(results_A_FF_all, results_A_FF[:,η_new_index])
        results_A_FF_all = results_A_FF_all[:, sortperm(results_A_FF_all[1,:], rev = true)]

        @load "results_eta_0.3_0.4_step_0.02.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF
        η_already = results_A_NFF_all[1,:]
        η_new = results_A_NFF[1,:]
        η_new_index = []
        for η_new_i in 1:length(η_new)
            if all(η_already .!= η_new[η_new_i])
                append!(η_new_index, [η_new_i])
            end
        end
        results_A_NFF_all = hcat(results_A_NFF_all, results_A_NFF[:,η_new_index])
        results_A_NFF_all = results_A_NFF_all[:, sortperm(results_A_NFF_all[1,:], rev = true)]
        results_A_FF_all = hcat(results_A_FF_all, results_A_FF[:,η_new_index])
        results_A_FF_all = results_A_FF_all[:, sortperm(results_A_FF_all[1,:], rev = true)]

        @save "results_eta_all.jld2" var_names results_A_NFF_all results_A_FF_all

    end

    #=======================#
    # solve transition path #
    #=======================#
    @load "results_eta_all.jld2" var_names results_A_NFF_all results_A_FF_all

    # extract all wage garnishment rates and the associated incentive multiplier
    η_all, λ_all = results_A_FF_all[1,:], results_A_FF_all[3,:]
    λ_all = λ_all[findall(η_all .!= 0.34)]
    η_all = η_all[findall(η_all .!= 0.34)] # η = 0.34 fails to converge. Why?
    η_benchmark = 0.25
    η_benchmark_index = findall(η_all .== η_benchmark)[]
    parameters_benchmark = parameters_function(η = η_benchmark)
    variables_benchmark = variables_function(parameters_benchmark; λ = λ_all[η_benchmark_index])
    solve_economy_function!(variables_benchmark, parameters_benchmark)

    # set parameters for the computation of transtion path
    T_size = 80
    T_degree = 15.0
    iter_max = 500
    tol = 1E-8
    slow_updating_transitional_dynamics = 0.1
    load_initial_value = true
    if load_initial_value == true
        @load "results_transition_eta_all.jld2" η_all λ_all transition_path_eta_all welfare_CEV_all_good_with_debt welfare_CEV_all_good_without_debt welfare_CEV_all_good welfare_CEV_all_bad welfare_CEV_all welfare_favor_all_good_with_debt welfare_favor_all_good_without_debt welfare_favor_all_good welfare_favor_all_bad welfare_favor_all
    else
        λ_all = λ_all[findall(η_all .!= η_benchmark)]
        η_all = η_all[findall(η_all .!= η_benchmark)]
        transition_path_eta_all = zeros(T_size+2)
        welfare_CEV_all_good_with_debt = zeros(1)
        welfare_CEV_all_good_without_debt = zeros(1)
        welfare_CEV_all_good = zeros(1)
        welfare_CEV_all_bad = zeros(1)
        welfare_CEV_all = zeros(1)
        welfare_favor_all_good_with_debt = zeros(1)
        welfare_favor_all_good_without_debt = zeros(1)
        welfare_favor_all_good = zeros(1)
        welfare_favor_all_bad = zeros(1)
        welfare_favor_all = zeros(1)
    end
    initial_z = ones(T_size+2)

    # solve transtion path from benchmark to all cases
    for η_i in 1:length(η_all)

        # solve the equilibrium with the new policy
        println("Solving steady state with η = $(η_all[η_i])...")
        parameters_new = parameters_function(η = η_all[η_i])
        variables_new = variables_function(parameters_new; λ = λ_all[η_i])
        solve_economy_function!(variables_new, parameters_new)

        # solve the transtion path
        println("Solving transitions from η = $η_benchmark to η = $(η_all[η_i])...")
        if load_initial_value == true
            variables_T = variables_T_function(transition_path_eta_all[:,η_i], initial_z, variables_benchmark, variables_new, parameters_new)
        else
            variables_T = variables_T_function(variables_benchmark, variables_new, parameters_new; T_size = T_size, T_degree = T_degree)
        end
        transitional_dynamic_λ_function!(variables_T, variables_benchmark, variables_new, parameters_new; tol = tol, iter_max = iter_max, slow_updating = slow_updating_transitional_dynamics)
        transition_path_eta = variables_T.aggregate_prices.leverage_ratio_λ
        plot_transtion_path_eta = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "", xlabel = "Time")
        plot_transtion_path_eta = plot!(transition_path_eta, linecolor = :blue, linewidth = 3, legend=:none)
        plot_transtion_path_eta
        Plots.savefig(plot_transtion_path_eta, pwd() * "\\figures\\transition path\\eta\\plot_transtion_path_eta_$(η_benchmark)_$(η_all[η_i]).pdf")

        # update the converged transition path of banking leverage ratio and compute welfare
        if load_initial_value == true
            
            transition_path_eta_all[:,η_i] = transition_path_eta
            
            welfare_CEV_all_good_with_debt[η_i] = 100 * sum(((variables_T.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,2] ./ variables_benchmark.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1])))
            
            welfare_CEV_all_good_without_debt[η_i] = 100 * sum(((variables_T.V[parameters_benchmark.a_ind_zero:end,:,:,:,:,2] ./ variables_benchmark.V[parameters_benchmark.a_ind_zero:end,:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1])))
            
            welfare_CEV_all_good[η_i] = 100 * sum(((variables_T.V[:,:,:,:,:,2] ./ variables_benchmark.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[:,:,:,:,:,1] ./ sum(variables_benchmark.μ[:,:,:,:,:,1])))
            
            welfare_CEV_all_bad[η_i] =  100 * sum(((variables_T.V_pos[:,:,:,:,:,2] ./ variables_benchmark.V_pos) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2])))

            welfare_CEV_all[η_i] = 100 * (sum(((variables_T.V[:,:,:,:,:,2] ./ variables_benchmark.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* variables_benchmark.μ[:,:,:,:,:,1]) + sum(((variables_T.V_pos[:,:,:,:,:,2] ./ variables_benchmark.V_pos) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2]))
            
            welfare_favor_all_good_with_debt[η_i] = 100 * sum((variables_T.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,2] .>= variables_benchmark.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:]) .* (variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1])))
            
            welfare_favor_all_good_without_debt[η_i] = 100 * sum((variables_T.V[parameters_benchmark.a_ind_zero:end,:,:,:,:,2] .>= variables_benchmark.V[parameters_benchmark.a_ind_zero:end,:,:,:,:]) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1])))
            
            welfare_favor_all_good[η_i] = 100 * sum((variables_T.V[:,:,:,:,:,2] .>= variables_benchmark.V) .* (variables_benchmark.μ[:,:,:,:,:,1] ./ sum(variables_benchmark.μ[:,:,:,:,:,1])))
            
            welfare_favor_all_bad[η_i] = 100 * sum((variables_T.V_pos[:,:,:,:,:,2] .>= variables_benchmark.V_pos) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2])))

            welfare_favor_all[η_i] = 100 * (sum((variables_T.V[:,:,:,:,:,2] .>= variables_benchmark.V) .* variables_benchmark.μ[:,:,:,:,:,1]) + sum((variables_T.V_pos[:,:,:,:,:,2] .> variables_benchmark.V_pos) .* variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2]))

        else

            transition_path_eta_all = hcat(transition_path_eta_all, transition_path_eta)
            
            welfare_CEV_all_good_with_debt = vcat(welfare_CEV_all_good_with_debt, 100 * sum(((variables_T.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,2] ./ variables_benchmark.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1]))))
            
            welfare_CEV_all_good_without_debt = vcat(welfare_CEV_all_good_without_debt, 100 * sum(((variables_T.V[parameters_benchmark.a_ind_zero:end,:,:,:,:,2] ./ variables_benchmark.V[parameters_benchmark.a_ind_zero:end,:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1]))))
            
            welfare_CEV_all_good = vcat(welfare_CEV_all_good, 100 * sum(((variables_T.V[:,:,:,:,:,2] ./ variables_benchmark.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[:,:,:,:,:,1] ./ sum(variables_benchmark.μ[:,:,:,:,:,1]))))
            
            welfare_CEV_all_bad = vcat(welfare_CEV_all_bad, 100 * sum(((variables_T.V_pos[:,:,:,:,:,2] ./ variables_benchmark.V_pos) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2]))))

            welfare_CEV_all = vcat(welfare_CEV_all, 100 * (sum(((variables_T.V[:,:,:,:,:,2] ./ variables_benchmark.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* variables_benchmark.μ[:,:,:,:,:,1]) + sum(((variables_T.V_pos[:,:,:,:,:,2] ./ variables_benchmark.V_pos) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2])))
            
            welfare_favor_all_good_with_debt = vcat(welfare_favor_all_good_with_debt, 100 * sum((variables_T.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,2] .>= variables_benchmark.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:]) .* (variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1]))))
            
            welfare_favor_all_good_without_debt = vcat(welfare_favor_all_good_without_debt, 100 * sum((variables_T.V[parameters_benchmark.a_ind_zero:end,:,:,:,:,2] .>= variables_benchmark.V[parameters_benchmark.a_ind_zero:end,:,:,:,:]) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1]))))
            
            welfare_favor_all_good = vcat(welfare_favor_all_good, 100 * sum((variables_T.V[:,:,:,:,:,2] .>= variables_benchmark.V) .* (variables_benchmark.μ[:,:,:,:,:,1] ./ sum(variables_benchmark.μ[:,:,:,:,:,1]))))
            
            welfare_favor_all_bad = vcat(welfare_favor_all_bad, 100 * sum((variables_T.V_pos[:,:,:,:,:,2] .>= variables_benchmark.V_pos) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2]))))

            welfare_favor_all = vcat(welfare_favor_all,  100 * (sum((variables_T.V[:,:,:,:,:,2] .>= variables_benchmark.V) .* variables_benchmark.μ[:,:,:,:,:,1]) + sum((variables_T.V_pos[:,:,:,:,:,2] .> variables_benchmark.V_pos) .* variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2])))

        end
    end

    # save results of transition path for all cases
    if load_initial_value == false
        transition_path_eta_all = transition_path_eta_all[:,2:end]
        welfare_CEV_all_good_with_debt = welfare_CEV_all_good_with_debt[2:end]
        welfare_CEV_all_good_without_debt = welfare_CEV_all_good_without_debt[2:end]
        welfare_CEV_all_good = welfare_CEV_all_good[2:end]
        welfare_CEV_all_bad = welfare_CEV_all_bad[2:end]
        welfare_CEV_all = welfare_CEV_all[2:end]
        welfare_favor_all_good_with_debt = welfare_favor_all_good_with_debt[2:end]
        welfare_favor_all_good_without_debt = welfare_favor_all_good_without_debt[2:end]
        welfare_favor_all_good = welfare_favor_all_good[2:end]
        welfare_favor_all_bad = welfare_favor_all_bad[2:end]
        welfare_favor_all = welfare_favor_all[2:end]
    end
    @save "results_transition_eta_all.jld2" η_all λ_all transition_path_eta_all welfare_CEV_all_good_with_debt welfare_CEV_all_good_without_debt welfare_CEV_all_good welfare_CEV_all_bad welfare_CEV_all welfare_favor_all_good_with_debt welfare_favor_all_good_without_debt welfare_favor_all_good welfare_favor_all_bad welfare_favor_all

    #=============================#
    # without financial frictions #
    #=============================#
    # create containers
    λ_all_NFF = zeros(length(η_all))
    welfare_CEV_all_good_with_debt_NFF = zeros(length(λ_all))
    welfare_CEV_all_good_without_debt_NFF = zeros(length(λ_all))
    welfare_CEV_all_good_NFF = zeros(length(λ_all))
    welfare_CEV_all_bad_NFF = zeros(length(λ_all))
    welfare_CEV_all_NFF = zeros(length(λ_all))
    welfare_favor_all_good_with_debt_NFF = zeros(length(λ_all))
    welfare_favor_all_good_without_debt_NFF = zeros(length(λ_all))
    welfare_favor_all_good_NFF = zeros(length(λ_all))
    welfare_favor_all_bad_NFF = zeros(length(λ_all))
    welfare_favor_all_NFF = zeros(length(λ_all))

    # run the benchmark results
    parameters_benchmark = parameters_function(η = η_benchmark)
    variables_benchmark = variables_function(parameters_benchmark; λ = 0.0)
    solve_economy_function!(variables_benchmark, parameters_benchmark)

    for η_i in 1:length(η_all)

        # solve the equilibrium with the new policy
        println("Solving steady state with η = $(η_all[η_i])...")
        parameters_new = parameters_function(η = η_all[η_i])
        variables_new = variables_function(parameters_new; λ = λ_all_NFF[η_i])
        solve_economy_function!(variables_new, parameters_new)

        # compute welfare
        welfare_CEV_all_good_with_debt_NFF[η_i] = 100 * sum(((variables_new.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:] ./ variables_benchmark.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1])))
        
        welfare_CEV_all_good_without_debt_NFF[η_i] = 100 * sum(((variables_new.V[parameters_benchmark.a_ind_zero:end,:,:,:,:] ./ variables_benchmark.V[parameters_benchmark.a_ind_zero:end,:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1])))
        
        welfare_CEV_all_good_NFF[η_i] = 100 * sum(((variables_new.V[:,:,:,:,:] ./ variables_benchmark.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[:,:,:,:,:,1] ./ sum(variables_benchmark.μ[:,:,:,:,:,1])))
        
        welfare_CEV_all_bad_NFF[η_i] =  100 * sum(((variables_new.V_pos[:,:,:,:,:] ./ variables_benchmark.V_pos) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2])))

        welfare_CEV_all_NFF[η_i] = 100 * (sum(((variables_new.V[:,:,:,:,:] ./ variables_benchmark.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* variables_benchmark.μ[:,:,:,:,:,1]) + sum(((variables_new.V_pos[:,:,:,:,:] ./ variables_benchmark.V_pos) .^ (1.0/(1.0-parameters_benchmark.σ)) .- 1.0) .* variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2]))
        
        welfare_favor_all_good_with_debt_NFF[η_i] = 100 * sum((variables_new.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:] .>= variables_benchmark.V[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:]) .* (variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_benchmark.μ[1:(parameters_benchmark.a_ind_zero-1),:,:,:,:,1])))
        
        welfare_favor_all_good_without_debt_NFF[η_i] = 100 * sum((variables_new.V[parameters_benchmark.a_ind_zero:end,:,:,:,:] .>= variables_benchmark.V[parameters_benchmark.a_ind_zero:end,:,:,:,:]) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,1])))
        
        welfare_favor_all_good_NFF[η_i] = 100 * sum((variables_new.V[:,:,:,:,:] .>= variables_benchmark.V) .* (variables_benchmark.μ[:,:,:,:,:,1] ./ sum(variables_benchmark.μ[:,:,:,:,:,1])))
        
        welfare_favor_all_bad_NFF[η_i] = 100 * sum((variables_new.V_pos[:,:,:,:,:] .>= variables_benchmark.V_pos) .* (variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2])))

        welfare_favor_all_NFF[η_i] = 100 * (sum((variables_new.V[:,:,:,:,:] .>= variables_benchmark.V) .* variables_benchmark.μ[:,:,:,:,:,1]) + sum((variables_new.V_pos[:,:,:,:,:] .> variables_benchmark.V_pos) .* variables_benchmark.μ[parameters_benchmark.a_ind_zero:end,:,:,:,:,2]))
    end    

    # save results
    @save "results_welfare_eta_all_NFF.jld2" η_all λ_all_NFF welfare_CEV_all_good_with_debt_NFF welfare_CEV_all_good_without_debt_NFF welfare_CEV_all_good_NFF welfare_CEV_all_bad_NFF welfare_CEV_all_NFF welfare_favor_all_good_with_debt_NFF welfare_favor_all_good_without_debt_NFF welfare_favor_all_good_NFF welfare_favor_all_bad_NFF welfare_favor_all_NFF

    # plot welfare results across η
    η_all = vcat(η_all, η_benchmark)
    welfare_CEV_all = vcat(welfare_CEV_all, 0.0)
    welfare_favor_all = vcat(welfare_favor_all, 100.0)
    welfare_CEV_all_NFF = vcat(welfare_CEV_all_NFF, 0.0)
    welfare_favor_all_NFF = vcat(welfare_favor_all_NFF, 100.0)

    welfare_CEV_all = welfare_CEV_all[sortperm(η_all)]
    welfare_favor_all = welfare_favor_all[sortperm(η_all)]
    welfare_CEV_all_NFF = welfare_CEV_all_NFF[sortperm(η_all)]
    welfare_favor_all_NFF = welfare_favor_all_NFF[sortperm(η_all)]
    η_all = η_all[sortperm(η_all)]

    plot_welfare_CEV_all = plot(size = (800,500), box = :on, legend = :topleft, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "%", xlabel = "\$ \\eta \$")
    plot_welfare_CEV_all = plot!(η_all, welfare_CEV_all, linecolor = :blue, linewidth = 3, label = "\$ \\textrm{with\\ financial\\ frictions} \$")
    plot_welfare_CEV_all = plot!(η_all, welfare_CEV_all_NFF, linecolor = :red, linewidth = 3, linestyle = :dash, label = "\$ \\textrm{without\\ financial\\ frictions} \$")
    plot_welfare_CEV_all
    Plots.savefig(plot_welfare_CEV_all, pwd() * "\\figures\\transition path\\eta\\plot_welfare_CEV_all.pdf")

    plot_welfare_favor_all = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "%", xlabel = "\$ \\eta \$")
    plot_welfare_favor_all = plot!(η_all, welfare_favor_all, linecolor = :blue, linewidth = 3, label = "\$ \\textrm{with\\ financial\\ frictions} \$")
    plot_welfare_favor_all = plot!(η_all, welfare_favor_all_NFF, linecolor = :red, linewidth = 3, linestyle = :dash, label = "\$ \\textrm{without\\ financial\\ frictions} \$")
    plot_welfare_favor_all
    Plots.savefig(plot_welfare_favor_all, pwd() * "\\figures\\transition path\\eta\\plot_welfare_favor_all.pdf")

end

if Indicator_solve_transitional_dynamics_across_p_h == true

    @load "results_p_h.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

    # specily the old and new policies
    p_h_8, λ_8 = results_A_FF[1,3], results_A_FF[3,3] # p_h = 1 / 8
    p_h_10, λ_10 = results_A_FF[1,2], results_A_FF[3,2] # p_h = 1 / 10
    p_h_12, λ_12 = results_A_FF[1,1], results_A_FF[3,1] # p_h = 1 / 12

    # stationary equilibrium when p_h = 1 / 8
    println("Solving steady state when p_h = $p_h_8...")
    parameters_8 = parameters_function(p_h = p_h_8)
    variables_8 = variables_function(parameters_8; λ = λ_8)
    solve_economy_function!(variables_8, parameters_8)
    # variables_8_NFF = variables_function(parameters_8; λ = 0.0)
    # solve_economy_function!(variables_8_NFF, parameters_8)

    # stationary equilibrium when p_h = 1 / 10
    println("Solving steady state when p_h = $p_h_10...")
    parameters_10 = parameters_function(p_h = p_h_10)
    variables_10 = variables_function(parameters_10; λ = λ_10)
    solve_economy_function!(variables_10, parameters_10)
    # variables_10_NFF = variables_function(parameters_10; λ = 0.0)
    # solve_economy_function!(variables_10_NFF, parameters_10)

    # stationary equilibrium when p_h = 1 / 12
    println("Solving steady state when p_h = $p_h_12...")
    parameters_12 = parameters_function(p_h = p_h_12)
    variables_12 = variables_function(parameters_12; λ = λ_12)
    solve_economy_function!(variables_12, parameters_12)
    # variables_12_NFF = variables_function(parameters_12; λ = 0.0)
    # solve_economy_function!(variables_12_NFF, parameters_12)

    # printout results of aggregate statistics
    results_equilibria_across_p_h = Any[
        "" "p_h = 1/8" "p_h = 1/10" "p_h = 1/12"
        "" "" "" ""
        "Banking leverage ratio" variables_8.aggregate_variables.leverage_ratio variables_10.aggregate_variables.leverage_ratio variables_12.aggregate_variables.leverage_ratio
        "Leverage premium (%)" variables_8.aggregate_prices.ι_λ*100 variables_10.aggregate_prices.ι_λ*100 variables_12.aggregate_prices.ι_λ*100
        "Wage" variables_8.aggregate_prices.w_λ variables_10.aggregate_prices.w_λ variables_12.aggregate_prices.w_λ
        "" "" "" ""
        "Default rate (%)" variables_8.aggregate_variables.share_of_filers*100 variables_10.aggregate_variables.share_of_filers*100 variables_12.aggregate_variables.share_of_filers*100
        "Share in debt (%)" variables_8.aggregate_variables.share_in_debts*100 variables_10.aggregate_variables.share_in_debts*100 variables_12.aggregate_variables.share_in_debts*100
        "Debt-to-earnings ratio (%)" variables_8.aggregate_variables.debt_to_earning_ratio*100 variables_10.aggregate_variables.debt_to_earning_ratio*100 variables_12.aggregate_variables.debt_to_earning_ratio*100
        "Average interest rate (%)" variables_8.aggregate_variables.avg_loan_rate*100 variables_10.aggregate_variables.avg_loan_rate*100 variables_12.aggregate_variables.avg_loan_rate*100
    ]
    display(results_equilibria_across_p_h)

    # save results
    CSV.write("results_equilibria_across_p_h.csv", Tables.table(results_equilibria_across_p_h), header = false)

    # set parameters for computation
    load_initial_value = true
    if load_initial_value == true
        @load "results_transition_p_h.jld2" transtion_path_p_h_10_12 transtion_path_p_h_10_8
    end
    T_size = 80
    T_degree = 15.0
    iter_max = 1000
    tol = 1E-8
    slow_updating_transitional_dynamics = 0.1
    initial_z = ones(T_size+2)

    # from p_h = 1 / 10 to p_h = 1 / 12
    println("Solving transitions from p_h = $p_h_10 to p_h = $p_h_12...")
    if load_initial_value == true
        variables_T_10_12 = variables_T_function(transtion_path_p_h_10_12, initial_z, variables_10, variables_12, parameters_12)
    else
        variables_T_10_12 = variables_T_function(variables_10, variables_12, parameters_12; T_size = T_size, T_degree = T_degree)
    end
    transitional_dynamic_λ_function!(variables_T_10_12, variables_10, variables_12, parameters_12; tol = tol, iter_max = iter_max, slow_updating = slow_updating_transitional_dynamics)
    transtion_path_p_h_10_12 = variables_T_10_12.aggregate_prices.leverage_ratio_λ
    plot_transtion_path_p_h_10_12 = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "", xlabel = "Time")
    plot_transtion_path_p_h_10_12 = plot!(transtion_path_p_h_10_12, linecolor = :blue, linewidth = 3, legend=:none)
    plot_transtion_path_p_h_10_12
    Plots.savefig(plot_transtion_path_p_h_10_12, pwd() * "\\figures\\plot_transtion_path_p_h_10_12.pdf")

    # from p_h = 1 / 10 to p_h = 1 / 8
    println("Solving transitions from p_h = $p_h_10 to p_h = $p_h_8...")
    if load_initial_value == true
        variables_T_10_8 = variables_T_function(transtion_path_p_h_10_8, initial_z, variables_10, variables_8, parameters_8)
    else
        variables_T_10_8 = variables_T_function(variables_10, variables_8, parameters_8; T_size = T_size, T_degree = T_degree)
    end
    transitional_dynamic_λ_function!(variables_T_10_8, variables_10, variables_8, parameters_8; tol = tol, iter_max = iter_max, slow_updating = slow_updating_transitional_dynamics)
    transtion_path_p_h_10_8 = variables_T_10_8.aggregate_prices.leverage_ratio_λ
    plot_transtion_path_p_h_10_8 = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "", xlabel = "Time")
    plot_transtion_path_p_h_10_8 = plot!(transtion_path_p_h_10_8, linecolor = :blue, linewidth = 3, legend=:none)
    plot_transtion_path_p_h_10_8
    Plots.savefig(plot_transtion_path_p_h_10_8, pwd() * "\\figures\\plot_transtion_path_p_h_10_8.pdf")

    # save transition path
    @save "results_transition_p_h.jld2" transtion_path_p_h_10_12 transtion_path_p_h_10_8

    # compute welfare metrics from p_h = 1/10 to p_h = 1/12
    welfare_CEV_10_12_good_with_debt = 100 * sum(((variables_T_10_12.V[1:(parameters_10.a_ind_zero-1),:,:,:,:,2] ./ variables_10.V[1:(parameters_10.a_ind_zero-1),:,:,:,:]) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* (variables_10.μ[1:(parameters_10.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_10.μ[1:(parameters_10.a_ind_zero-1),:,:,:,:,1])))
    welfare_CEV_10_12_good_no_debt = 100 * sum(((variables_T_10_12.V[parameters_10.a_ind_zero:end,:,:,:,:,2] ./ variables_10.V[parameters_10.a_ind_zero:end,:,:,:,:]) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* (variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,1])))
    welfare_CEV_10_12_good = 100 * sum(((variables_T_10_12.V[:,:,:,:,:,2] ./ variables_10.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* (variables_10.μ[:,:,:,:,:,1] ./ sum(variables_10.μ[:,:,:,:,:,1])))
    welfare_CEV_10_12_bad =  100 * sum(((variables_T_10_12.V_pos[:,:,:,:,:,2] ./ variables_10.V_pos) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* (variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2])))
    welfare_CEV_10_12 = 100 * (sum(((variables_T_10_12.V[:,:,:,:,:,2] ./ variables_10.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* variables_10.μ[:,:,:,:,:,1]) + sum(((variables_T_10_12.V_pos[:,:,:,:,:,2] ./ variables_10.V_pos) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2]))

    welfare_favor_10_12_good_with_debt = 100 * sum((variables_T_10_12.V[1:(parameters_10.a_ind_zero-1),:,:,:,:,2] .>= variables_10.V[1:(parameters_10.a_ind_zero-1),:,:,:,:]) .* (variables_10.μ[1:(parameters_10.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_10.μ[1:(parameters_10.a_ind_zero-1),:,:,:,:,1])))
    welfare_favor_10_12_good_without_debt = 100 * sum((variables_T_10_12.V[parameters_10.a_ind_zero:end,:,:,:,:,2] .>= variables_10.V[parameters_10.a_ind_zero:end,:,:,:,:]) .* (variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,1])))
    welfare_favor_10_12_good = 100 * sum((variables_T_10_12.V[:,:,:,:,:,2] .>= variables_10.V) .* (variables_10.μ[:,:,:,:,:,1] ./ sum(variables_10.μ[:,:,:,:,:,1])))
    welfare_favor_10_12_bad = 100 * sum((variables_T_10_12.V_pos[:,:,:,:,:,2] .>= variables_10.V_pos) .* (variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2])))
    welfare_favor_10_12 = 100 * (sum((variables_T_10_12.V[:,:,:,:,:,2] .>= variables_10.V) .* variables_10.μ[:,:,:,:,:,1]) + sum((variables_T_10_12.V_pos[:,:,:,:,:,2] .>= variables_10.V_pos) .* variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2]))

    # compute welfare metrics from η = 0.25 to η = 0.20
    welfare_CEV_10_8_good_with_debt = 100 * sum(((variables_T_10_8.V[1:(parameters_10.a_ind_zero-1),:,:,:,:,2] ./ variables_10.V[1:(parameters_10.a_ind_zero-1),:,:,:,:]) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* (variables_10.μ[1:(parameters_10.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_10.μ[1:(parameters_10.a_ind_zero-1),:,:,:,:,1])))
    welfare_CEV_10_8_good_no_debt = 100 * sum(((variables_T_10_8.V[parameters_10.a_ind_zero:end,:,:,:,:,2] ./ variables_10.V[parameters_10.a_ind_zero:end,:,:,:,:]) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* (variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,1])))
    welfare_CEV_10_8_good = 100 * sum(((variables_T_10_8.V[:,:,:,:,:,2] ./ variables_10.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* (variables_10.μ[:,:,:,:,:,1] ./ sum(variables_10.μ[:,:,:,:,:,1])))
    welfare_CEV_10_8_bad =  100 * sum(((variables_T_10_8.V_pos[:,:,:,:,:,2] ./ variables_10.V_pos) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* (variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2])))
    welfare_CEV_10_8 = 100 * (sum(((variables_T_10_8.V[:,:,:,:,:,2] ./ variables_10.V[:,:,:,:,:]) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* variables_10.μ[:,:,:,:,:,1]) + sum(((variables_T_10_8.V_pos[:,:,:,:,:,2] ./ variables_10.V_pos) .^ (1.0/(1.0-parameters_10.σ)) .- 1.0) .* variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2]))

    welfare_favor_10_8_good_with_debt = 100 * sum((variables_T_10_8.V[1:(parameters_10.a_ind_zero-1),:,:,:,:,2] .>= variables_10.V[1:(parameters_10.a_ind_zero-1),:,:,:,:]) .* (variables_10.μ[1:(parameters_10.a_ind_zero-1),:,:,:,:,1] ./ sum(variables_10.μ[1:(parameters_10.a_ind_zero-1),:,:,:,:,1])))
    welfare_favor_10_8_good_without_debt = 100 * sum((variables_T_10_8.V[parameters_10.a_ind_zero:end,:,:,:,:,2] .>= variables_10.V[parameters_10.a_ind_zero:end,:,:,:,:]) .* (variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,1] ./ sum(variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,1])))
    welfare_favor_10_8_good = 100 * sum((variables_T_10_8.V[:,:,:,:,:,2] .>= variables_10.V) .* (variables_10.μ[:,:,:,:,:,1] ./ sum(variables_10.μ[:,:,:,:,:,1])))
    welfare_favor_10_8_bad = 100* sum((variables_T_10_8.V_pos[:,:,:,:,:,2] .>= variables_10.V_pos) .* (variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2] ./ sum(variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2])))
    welfare_favor_10_8 = 100 * (sum((variables_T_10_8.V[:,:,:,:,:,2] .>= variables_10.V) .* variables_10.μ[:,:,:,:,:,1]) + sum((variables_T_10_8.V_pos[:,:,:,:,:,2] .>= variables_10.V_pos) .* variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,2]))

    # share of households
    HHs_good_debt = 100 * sum(variables_10.μ[1:(parameters_10.a_ind_zero-1),:,:,:,:,1])
    HHs_good_no_debt = 100 * sum(variables_10.μ[parameters_10.a_ind_zero:end,:,:,:,:,1])
    HHs_good = HHs_good_debt + HHs_good_no_debt
    HHs_good_debt_cond = HHs_good_debt / HHs_good * 100
    HHs_good_no_debt_cond = HHs_good_no_debt / HHs_good * 100
    HHs_bad = 100 * sum(variables_10.μ[:,:,:,:,:,2])
    HHs_total = HHs_good + HHs_bad

    # printout results of welfare effects
    results_welfare_across_p_h = Any[
        "" "" "from p_h = 1/10 to 1/8" "" "from p_h = 1/10 to 1/12" ""
        "(%)" "Proportion" "C.E.V." "Favor Reform" "C.E.V." "Favor Reform"
        "" "" "" "" "" ""
        "Have good credit history" HHs_good welfare_CEV_10_8_good welfare_favor_10_8_good welfare_CEV_10_12_good welfare_favor_10_12_good
        "Indebted" HHs_good_debt_cond welfare_CEV_10_8_good_with_debt welfare_favor_10_8_good_with_debt welfare_CEV_10_12_good_with_debt welfare_favor_10_12_good_with_debt
        "Not indebted" HHs_good_no_debt_cond welfare_CEV_10_8_good_no_debt welfare_favor_10_8_good_without_debt welfare_CEV_10_12_good_no_debt welfare_favor_10_12_good_without_debt
        "" "" "" "" "" ""
        "Have bad credit history" HHs_bad welfare_CEV_10_8_bad welfare_favor_10_8_bad welfare_CEV_10_12_bad welfare_favor_10_12_bad
        "" "" "" "" "" ""
        "Total" HHs_total welfare_CEV_10_8 welfare_favor_10_8 welfare_CEV_10_12 welfare_favor_10_12
    ]
    display(results_welfare_across_p_h)

    # save results
    CSV.write("results_welfare_across_p_h.csv", Tables.table(results_welfare_across_p_h), header = false)

end

#================#
# MIT shock to z #
#================#

if Indicator_solve_transitional_dynamics_MIT_z == true

    #=================#
    # solve benchmark #
    #=================#
    @load "results_eta_all.jld2" var_names results_A_NFF_all results_A_FF_all
    η_all, λ_all = results_A_FF_all[1,:], results_A_FF_all[3,:]
    η_benchmark = 0.25
    η_benchmark_index = findall(η_all .== η_benchmark)[]
    parameters_benchmark = parameters_function(η = η_benchmark)
    variables_benchmark = variables_function(parameters_benchmark; λ = λ_all[η_benchmark_index])
    solve_economy_function!(variables_benchmark, parameters_benchmark)

    #===========================#
    # solve the transition path #
    #===========================#
    # set parameters for computation
    T_size = 80
    T_degree = 15.0
    iter_max = 1000
    tol = 1E-8
    slow_updating_transitional_dynamics = 0.1

    # set up transition path of MIT shock to z
    ρ_z = 0.85
    σ_z = 0.01
    path_z_negative = zeros(T_size+2)
    path_z_positive = zeros(T_size+2)
    for t in 2:(T_size+1)
        path_z_negative[t] = t == 2 ? -σ_z : ρ_z * path_z_negative[t-1]
        path_z_positive[t] = t == 2 ? σ_z : ρ_z * path_z_positive[t-1]
    end
    path_z_negative = exp.(path_z_negative)
    path_z_positive = exp.(path_z_positive)

    # set up transition path of banking leverage ratio
    load_initial_value = false
    if load_initial_value == true
        @load "results_transition_MIT_z.jld2" transtion_path_MIT_z_negative transtion_path_MIT_z_positive
    else
        transtion_path_MIT_z_negative = path_z_positive .* variables_benchmark.aggregate_variables.leverage_ratio
        transtion_path_MIT_z_positive = path_z_negative .* variables_benchmark.aggregate_variables.leverage_ratio
    end

    # solve the transition path of banking leverage ratio if negative shock
    println("Solving transitions path for negative MIT shock to z")
    variables_T_MIT_z_negative = variables_T_function(transtion_path_MIT_z_negative, path_z_negative, variables_benchmark, variables_benchmark, parameters_benchmark)
    transitional_dynamic_λ_function!(variables_T_MIT_z_negative, variables_benchmark, variables_benchmark, parameters_benchmark; tol = tol, iter_max = iter_max, slow_updating = slow_updating_transitional_dynamics)

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
    age_max = convert(Int64, ceil(1.0 / (1.0 - parameters.ρ)))
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

    plot_consumption = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "", xlabel = "working age")
    plot_consumption = plot!(1:age_max, mean_consumption_age_FF, label="\$ \\theta = 1/3.0\\ (\\textrm{with\\ financial\\ frictions}) \$", linecolor = :blue, linewidth = 3)
    plot_consumption = plot!(1:age_max, mean_consumption_age_NFF, label="\$ \\theta = 0\\ (\\textrm{no\\ financial\\ frictions}) \$", linecolor = :red, linestyle = :dot, linewidth = 3)
    plot_consumption
    Plots.savefig(plot_consumption, pwd() * "\\figures\\plot_consumption.pdf")

    df = DataFrame(x = 1:age_max)
    df.y = (mean_consumption_age_NFF .- mean_consumption_age_FF) ./ mean_consumption_age_FF .* 100
    model = lm(@formula(y ~ 1 + x), df)
    plot_consumption_comparison = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "%", xlabel = "\$ \\textrm{working\\ age} \$")
    plot_consumption_comparison = plot!(df.x, df.y, linecolor = :blue, linewidth = 3, label=:none)
    plot_consumption_comparison = plot!(df.x, predict(model, df), linecolor = :blue, linestyle = :dot, linewidth = 3, label=:none)
    plot_consumption_comparison
    Plots.savefig(plot_consumption_comparison, pwd() * "\\figures\\plot_consumption_comparison.pdf")

    growth_consumption_age_FF = (mean_consumption_age_FF[2:end] .- mean_consumption_age_FF[1:(end-1)]) ./ mean_consumption_age_FF[1:(end-1)] * 100
    growth_consumption_age_NFF = (mean_consumption_age_NFF[2:end] .- mean_consumption_age_NFF[1:(end-1)]) ./ mean_consumption_age_NFF[1:(end-1)] * 100
    plot_consumption_growth = plot(size = (800,500), box = :on, legend = :topright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "%", xlabel = "\$ \\textrm{working\\ age} \$")
    plot_consumption_growth = plot!(2:age_max, growth_consumption_age_FF, label="\$ \\theta = 1/3.0\\ (\\textrm{with\\ financial\\ frictions}) \$", linecolor = :blue, linewidth = 3)
    plot_consumption_growth = plot!(2:age_max, growth_consumption_age_NFF, label="\$ \\theta = 0\\ (\\textrm{no\\ financial\\ frictions}) \$", linecolor = :red, linestyle = :dot, linewidth = 3)
    plot_consumption_growth
    Plots.savefig(plot_consumption_growth, pwd() * "\\figures\\plot_consumption_growth.pdf")

    plot_var_log_consumption = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "\$ \\textrm{variance\\ of\\ log\\ consumption} \$", xlabel = "\$ \\textrm{working\\ age} \$")
    plot_var_log_consumption = plot!(1:age_max, variance_log_consumption_age_FF, legend=:bottomright, label="\$ \\theta = 1/3\\ (\\textrm{with\\ financial\\ frictions}) \$", linecolor = :blue, linewidth = 3)
    plot_var_log_consumption = plot!(1:age_max, variance_log_consumption_age_NFF, label="\$ \\theta = 0\\ (\\textrm{no\\ financial\\ frictions}) \$", linecolor = :red, linestyle = :dot, linewidth = 3)
    plot_var_log_consumption
    Plots.savefig(plot_var_log_consumption, pwd() * "\\figures\\plot_var_log_consumption.pdf")

    df = DataFrame(x = 1:age_max)
    df.y = (variance_log_consumption_age_NFF .- variance_log_consumption_age_FF) ./ variance_log_consumption_age_FF .* 100
    model = lm(@formula(y ~ 1 + x), df)
    plot_var_log_consumption_comparison = plot(size = (800,500), box = :on, legend = :bottomleft, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "%", xlabel = "\$ \\textrm{working\\ age} \$")
    plot_var_log_consumption_comparison = plot!(df.x, df.y, linecolor = :blue, linewidth = 3, label=:none)
    plot_var_log_consumption_comparison = plot!(df.x, predict(model, df), linecolor = :blue, linestyle = :dot, linewidth = 3, label=:none)
    plot_var_log_consumption_comparison
    Plots.savefig(plot_var_log_consumption_comparison, pwd() * "\\figures\\plot_var_log_consumption_comparison.pdf")

end

if Indicator_simulation_across_θ == true

    # specify parameters
    num_hh = 40000
    num_periods = 5100
    burn_in = 100

    # load stationary equilibria across θ
    @load "results_theta.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

    # extract equilibria with financial fricitons
    θ_1, λ_1 = results_A_FF[1,1], results_A_FF[3,1]
    θ_2, λ_2 = results_A_FF[1,2], results_A_FF[3,2]
    θ_3, λ_3 = results_A_FF[1,3], results_A_FF[3,3]

    # stationary equilibria across θ with financial frictions
    println("Solving steady state with θ = $θ_1...")
    parameters_θ_1 = parameters_function(θ = θ_1)
    variables_θ_1 = variables_function(parameters_θ_1; λ = λ_1)
    solve_economy_function!(variables_θ_1, parameters_θ_1)
    println("Solving steady state with θ = $θ_2...")
    parameters_θ_2 = parameters_function(θ = θ_2)
    variables_θ_2 = variables_function(parameters_θ_2; λ = λ_2)
    solve_economy_function!(variables_θ_2, parameters_θ_2)
    println("Solving steady state with θ = $θ_3...")
    parameters_θ_3 = parameters_function(θ = θ_3)
    variables_θ_3 = variables_function(parameters_θ_3; λ = λ_3)
    solve_economy_function!(variables_θ_3, parameters_θ_3)

    # simulate models
    panel_asset_θ_1, panel_history_θ_1, panel_default_θ_1, panel_age_θ_1, panel_consumption_θ_1, shock_ρ_θ_1, shock_e_1_θ_1, shock_e_2_θ_1, shock_e_3_θ_1, shock_ν_θ_1 = simulation(variables_θ_1, parameters_θ_1; num_hh = num_hh, num_periods = num_periods, burn_in = burn_in)
    panel_asset_θ_2, panel_history_θ_2, panel_default_θ_2, panel_age_θ_2, panel_consumption_θ_2, shock_ρ_θ_2, shock_e_1_θ_2, shock_e_2_θ_2, shock_e_3_θ_2, shock_ν_θ_2 = simulation(variables_θ_2, parameters_θ_2; num_hh = num_hh, num_periods = num_periods, burn_in = burn_in)
    panel_asset_θ_3, panel_history_θ_3, panel_default_θ_3, panel_age_θ_3, panel_consumption_θ_3, shock_ρ_θ_3, shock_e_1_θ_3, shock_e_2_θ_3, shock_e_3_θ_3, shock_ν_θ_3 = simulation(variables_θ_3, parameters_θ_3; num_hh = num_hh, num_periods = num_periods, burn_in = burn_in)

    # save simulation results
    @save "simulations_benchmark_theta_1.jld2" panel_asset_θ_1 panel_history_θ_1 panel_default_θ_1 panel_age_θ_1 panel_consumption_θ_1 shock_ρ_θ_1 shock_e_1_θ_1 shock_e_2_θ_1 shock_e_3_θ_1 shock_ν_θ_1
    @save "simulations_benchmark_theta_2.jld2" panel_asset_θ_2 panel_history_θ_2 panel_default_θ_2 panel_age_θ_2 panel_consumption_θ_2 shock_ρ_θ_2 shock_e_1_θ_2 shock_e_2_θ_2 shock_e_3_θ_2 shock_ν_θ_2
    @save "simulations_benchmark_theta_3.jld2" panel_asset_θ_3 panel_history_θ_3 panel_default_θ_3 panel_age_θ_3 panel_consumption_θ_3 shock_ρ_θ_3 shock_e_1_θ_3 shock_e_2_θ_3 shock_e_3_θ_3 shock_ν_θ_3

end

if Indicator_simulation_across_θ_results == true

    # load simulation results
    @load "simulations_benchmark_NFF.jld2" panel_asset_NFF panel_history_NFF panel_default_NFF panel_age_NFF panel_consumption_NFF shock_ρ_NFF shock_e_1_NFF shock_e_2_NFF shock_e_3_NFF shock_ν_NFF
    @load "simulations_benchmark_theta_1.jld2" panel_asset_θ_1 panel_history_θ_1 panel_default_θ_1 panel_age_θ_1 panel_consumption_θ_1 shock_ρ_θ_1 shock_e_1_θ_1 shock_e_2_θ_1 shock_e_3_θ_1 shock_ν_θ_1
    @load "simulations_benchmark_theta_2.jld2" panel_asset_θ_2 panel_history_θ_2 panel_default_θ_2 panel_age_θ_2 panel_consumption_θ_2 shock_ρ_θ_2 shock_e_1_θ_2 shock_e_2_θ_2 shock_e_3_θ_2 shock_ν_θ_2
    @load "simulations_benchmark_theta_3.jld2" panel_asset_θ_3 panel_history_θ_3 panel_default_θ_3 panel_age_θ_3 panel_consumption_θ_3 shock_ρ_θ_3 shock_e_1_θ_3 shock_e_2_θ_3 shock_e_3_θ_3 shock_ν_θ_3

    # set parameters
    parameters = parameters_function()
    num_periods = size(panel_asset_θ_1)[2]
    num_hh = size(panel_asset_θ_1)[1]

    # share of defaulters
    fraction_default_sim_θ_1 = zeros(num_periods)
    fraction_default_sim_θ_2 = zeros(num_periods)
    fraction_default_sim_θ_3 = zeros(num_periods)
    for i in 1:num_periods
        fraction_default_sim_θ_1[i] = sum(panel_default_θ_1[:,i]) / num_hh * 100
        fraction_default_sim_θ_2[i] = sum(panel_default_θ_2[:,i]) / num_hh * 100
        fraction_default_sim_θ_3[i] = sum(panel_default_θ_3[:,i]) / num_hh * 100
    end
    fraction_default_sim_θ_1_avg = sum(fraction_default_sim_θ_1) / num_periods
    fraction_default_sim_θ_2_avg = sum(fraction_default_sim_θ_2) / num_periods
    fraction_default_sim_θ_3_avg = sum(fraction_default_sim_θ_3) / num_periods

    # share in debts
    fraction_debts_sim_θ_1 = zeros(num_periods)
    fraction_debts_sim_θ_2 = zeros(num_periods)
    fraction_debts_sim_θ_3 = zeros(num_periods)
    for i in 1:num_periods
        fraction_debts_sim_θ_1[i] = sum(panel_asset_θ_1[:,i] .< parameters.a_ind_zero) / num_hh * 100
        fraction_debts_sim_θ_2[i] = sum(panel_asset_θ_2[:,i] .< parameters.a_ind_zero) / num_hh * 100
        fraction_debts_sim_θ_3[i] = sum(panel_asset_θ_3[:,i] .< parameters.a_ind_zero) / num_hh * 100
    end
    fraction_debts_sim_θ_1_avg = sum(fraction_debts_sim_θ_1) / num_periods
    fraction_debts_sim_θ_2_avg = sum(fraction_debts_sim_θ_2) / num_periods
    fraction_debts_sim_θ_3_avg = sum(fraction_debts_sim_θ_3) / num_periods

    # share of defaulters, conditional borrowing
    fraction_cond_default_sim_θ_1_avg = fraction_default_sim_θ_1_avg / fraction_debts_sim_θ_1_avg * 100
    fraction_cond_default_sim_θ_2_avg = fraction_default_sim_θ_2_avg / fraction_debts_sim_θ_2_avg * 100
    fraction_cond_default_sim_θ_3_avg = fraction_default_sim_θ_3_avg / fraction_debts_sim_θ_3_avg * 100

    # consumption over life cycle
    age_max = convert(Int64, ceil(1.0 / (1.0 - parameters.ρ)))

    mean_consumption_age_NFF = zeros(age_max)
    variance_consumption_age_NFF = zeros(age_max)
    panel_log_consumption_NFF = log.(panel_consumption_NFF)
    mean_log_consumption_age_NFF = zeros(age_max)
    variance_log_consumption_age_NFF = zeros(age_max)

    mean_consumption_age_θ_1, mean_consumption_age_θ_2, mean_consumption_age_θ_3 = zeros(age_max), zeros(age_max), zeros(age_max)
    variance_consumption_age_θ_1, variance_consumption_age_θ_2, variance_consumption_age_θ_3 = zeros(age_max), zeros(age_max), zeros(age_max)
    panel_log_consumption_θ_1, panel_log_consumption_θ_2, panel_log_consumption_θ_3 = log.(panel_consumption_θ_1), log.(panel_consumption_θ_2), log.(panel_consumption_θ_3)
    mean_log_consumption_age_θ_1, mean_log_consumption_age_θ_2, mean_log_consumption_age_θ_3 = zeros(age_max), zeros(age_max), zeros(age_max)
    variance_log_consumption_age_θ_1, variance_log_consumption_age_θ_2, variance_log_consumption_age_θ_3 = zeros(age_max), zeros(age_max), zeros(age_max)

    for age_i in 1:age_max
        age_bool_NFF = (panel_age_NFF .== age_i)
        mean_consumption_age_NFF[age_i] = sum(panel_consumption_NFF[age_bool_NFF]) / sum(age_bool_NFF)
        variance_consumption_age_NFF[age_i] = sum((panel_consumption_NFF[age_bool_NFF] .- mean_consumption_age_NFF[age_i]).^2) / sum(age_bool_NFF)
        mean_log_consumption_age_NFF[age_i] = sum(panel_log_consumption_NFF[age_bool_NFF]) / sum(age_bool_NFF)
        variance_log_consumption_age_NFF[age_i] = sum((panel_log_consumption_NFF[age_bool_NFF] .- mean_log_consumption_age_NFF[age_i]).^2) / sum(age_bool_NFF)

        age_bool_θ_1 = (panel_age_θ_1 .== age_i)
        mean_consumption_age_θ_1[age_i] = sum(panel_consumption_θ_1[age_bool_θ_1]) / sum(age_bool_θ_1)
        variance_consumption_age_θ_1[age_i] = sum((panel_consumption_θ_1[age_bool_θ_1] .- mean_consumption_age_θ_1[age_i]).^2) / sum(age_bool_θ_1)
        mean_log_consumption_age_θ_1[age_i] = sum(panel_log_consumption_θ_1[age_bool_θ_1]) / sum(age_bool_θ_1)
        variance_log_consumption_age_θ_1[age_i] = sum((panel_log_consumption_θ_1[age_bool_θ_1] .- mean_log_consumption_age_θ_1[age_i]).^2) / sum(age_bool_θ_1)

        age_bool_θ_2 = (panel_age_θ_2 .== age_i)
        mean_consumption_age_θ_2[age_i] = sum(panel_consumption_θ_2[age_bool_θ_2]) / sum(age_bool_θ_2)
        variance_consumption_age_θ_2[age_i] = sum((panel_consumption_θ_2[age_bool_θ_2] .- mean_consumption_age_θ_2[age_i]).^2) / sum(age_bool_θ_2)
        mean_log_consumption_age_θ_2[age_i] = sum(panel_log_consumption_θ_2[age_bool_θ_2]) / sum(age_bool_θ_2)
        variance_log_consumption_age_θ_2[age_i] = sum((panel_log_consumption_θ_2[age_bool_θ_2] .- mean_log_consumption_age_θ_2[age_i]).^2) / sum(age_bool_θ_2)

        age_bool_θ_3 = (panel_age_θ_3 .== age_i)
        mean_consumption_age_θ_3[age_i] = sum(panel_consumption_θ_3[age_bool_θ_3]) / sum(age_bool_θ_3)
        variance_consumption_age_θ_3[age_i] = sum((panel_consumption_θ_3[age_bool_θ_3] .- mean_consumption_age_θ_3[age_i]).^2) / sum(age_bool_θ_3)
        mean_log_consumption_age_θ_3[age_i] = sum(panel_log_consumption_θ_3[age_bool_θ_3]) / sum(age_bool_θ_3)
        variance_log_consumption_age_θ_3[age_i] = sum((panel_log_consumption_θ_3[age_bool_θ_3] .- mean_log_consumption_age_θ_3[age_i]).^2) / sum(age_bool_θ_3)
    end

    plot_consumption_across_θ = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "consumption", xlabel = "working age")
    plot_consumption_across_θ = plot!(1:age_max, mean_consumption_age_θ_2, label="\$ \\theta = 1/3.0 \$", linecolor = :blue, linewidth = 3)
    plot_consumption_across_θ = plot!(1:age_max, mean_consumption_age_θ_1, label="\$ \\theta = 1/2.7 \$", linecolor = :red, linestyle = :dash, linewidth = 3)
    plot_consumption_across_θ = plot!(1:age_max, mean_consumption_age_θ_3, label="\$ \\theta = 1/3.3 \$", linecolor = :black, linestyle = :dashdot, linewidth = 3)
    plot_consumption_across_θ = plot!(1:age_max, mean_consumption_age_NFF, label="\$ \\theta = 0 \$", linecolor = :grey, linestyle = :dot, linewidth = 3)
    plot_consumption_across_θ
    Plots.savefig(plot_consumption_across_θ, pwd() * "\\figures\\plot_consumption_across_theta.pdf")

    plot_var_log_consumption_across_θ = plot(size = (800,500), box = :on, legend = :bottomright, xtickfont = font(18, "Computer Modern", :black), ytickfont = font(18, "Computer Modern", :black), titlefont = font(18, "Computer Modern", :black), guidefont = font(18, "Computer Modern", :black), legendfont = font(18, "Computer Modern", :black), margin = 4mm, ylabel = "variance of log consumption", xlabel = "working age")
    plot_var_log_consumption_across_θ = plot!(1:age_max, variance_log_consumption_age_θ_2, label="\$ \\theta = 1/3.0 \$", linecolor = :blue, linewidth = 3)
    plot_var_log_consumption_across_θ = plot!(1:age_max, variance_log_consumption_age_θ_1, label="\$ \\theta = 1/2.7 \$", linecolor = :red, linestyle = :dash, linewidth = 3)
    plot_var_log_consumption_across_θ = plot!(1:age_max, variance_log_consumption_age_θ_3, label="\$ \\theta = 1/3.3 \$", linecolor = :black, linestyle = :dashdot, linewidth = 3)
    plot_var_log_consumption_across_θ = plot!(1:age_max, variance_log_consumption_age_NFF, label="\$ \\theta = 0 \$", linecolor = :grey, linestyle = :dot, linewidth = 3)
    plot_var_log_consumption_across_θ
    Plots.savefig(plot_var_log_consumption_across_θ, pwd() * "\\figures\\plot_var_log_consumption_across_theta.pdf")

end

if Indicator_simulation_across_ψ == true

    # specify parameters
    num_hh = 40000
    num_periods = 5100
    burn_in = 100

    # load stationary equilibria across ψ
    @load "results_psi.jld2" var_names results_A_NFF results_V_NFF results_V_pos_NFF results_μ_NFF results_A_FF results_V_FF results_V_pos_FF results_μ_FF

    # extract equilibria with financial fricitons
    ψ_1, λ_1 = results_A_FF[1,1], results_A_FF[3,1]
    ψ_2, λ_2 = results_A_FF[1,2], results_A_FF[3,2]
    ψ_3, λ_3 = results_A_FF[1,3], results_A_FF[3,3]

    # stationary equilibria across ψ with financial frictions
    println("Solving steady state with ψ = $ψ_1...")
    parameters_ψ_1 = parameters_function(ψ = ψ_1)
    variables_ψ_1 = variables_function(parameters_ψ_1; λ = λ_1)
    solve_economy_function!(variables_ψ_1, parameters_ψ_1)
    println("Solving steady state with ψ = $ψ_2...")
    parameters_ψ_2 = parameters_function(ψ = ψ_2)
    variables_ψ_2 = variables_function(parameters_ψ_2; λ = λ_2)
    solve_economy_function!(variables_ψ_2, parameters_ψ_2)
    println("Solving steady state with ψ = $ψ_3...")
    parameters_ψ_3 = parameters_function(ψ = ψ_3)
    variables_ψ_3 = variables_function(parameters_ψ_3; λ = λ_3)
    solve_economy_function!(variables_ψ_3, parameters_ψ_3)

    # simulate models
    panel_asset_ψ_1, panel_history_ψ_1, panel_default_ψ_1, panel_age_ψ_1, panel_consumption_ψ_1, shock_ρ_ψ_1, shock_e_1_ψ_1, shock_e_2_ψ_1, shock_e_3_ψ_1, shock_ν_ψ_1 = simulation(variables_ψ_1, parameters_ψ_1; num_hh = num_hh, num_periods = num_periods, burn_in = burn_in)
    panel_asset_ψ_2, panel_history_ψ_2, panel_default_ψ_2, panel_age_ψ_2, panel_consumption_ψ_2, shock_ρ_ψ_2, shock_e_1_ψ_2, shock_e_2_ψ_2, shock_e_3_ψ_2, shock_ν_ψ_2 = simulation(variables_ψ_2, parameters_ψ_2; num_hh = num_hh, num_periods = num_periods, burn_in = burn_in)
    panel_asset_ψ_3, panel_history_ψ_3, panel_default_ψ_3, panel_age_ψ_3, panel_consumption_ψ_3, shock_ρ_ψ_3, shock_e_1_ψ_3, shock_e_2_ψ_3, shock_e_3_ψ_3, shock_ν_ψ_3 = simulation(variables_ψ_3, parameters_ψ_3; num_hh = num_hh, num_periods = num_periods, burn_in = burn_in)

    # save simulation results
    @save "simulations_benchmark_psi_1.jld2" panel_asset_ψ_1 panel_history_ψ_1 panel_default_ψ_1 panel_age_ψ_1 panel_consumption_ψ_1 shock_ρ_ψ_1 shock_e_1_ψ_1 shock_e_2_ψ_1 shock_e_3_ψ_1 shock_ν_ψ_1
    @save "simulations_benchmark_psi_2.jld2" panel_asset_ψ_2 panel_history_ψ_2 panel_default_ψ_2 panel_age_ψ_2 panel_consumption_ψ_2 shock_ρ_ψ_2 shock_e_1_ψ_2 shock_e_2_ψ_2 shock_e_3_ψ_2 shock_ν_ψ_2
    @save "simulations_benchmark_psi_3.jld2" panel_asset_ψ_3 panel_history_ψ_3 panel_default_ψ_3 panel_age_ψ_3 panel_consumption_ψ_3 shock_ρ_ψ_3 shock_e_1_ψ_3 shock_e_2_ψ_3 shock_e_3_ψ_3 shock_ν_ψ_3

end
