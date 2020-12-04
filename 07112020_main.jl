include("packages.jl")
include("functions_07112020.jl")
include("functions_07112020_MIT.jl")

#==============================#
# solve the steady-state value #
#==============================#
λ_optimal = 0.02160039620763443
println("Running Julia with $(Threads.nthreads()) threads...")
parameters = para_func(; λ = λ_optimal)
variables = var_func(parameters)
solve_func!(variables, parameters)

data_spec = Any[#= 1=# "Number of Endowment"                parameters.e_size;
                #= 2=# "Number of Assets"                   parameters.a_size;
                #= 3=# "Number of Negative Assets"          parameters.a_size_neg;
                #= 4=# "Number of Positive Assets"          parameters.a_size_pos;
                #= 5=# "Number of Assets (for Density)"     parameters.a_size_μ;
                #= 6=# "Minimum of Assets"                  parameters.a_grid[1];
                #= 7=# "Maximum of Assets"                  parameters.a_grid[end];
                #= 8=# "Scale of Impatience"                parameters.ν_grid[1];
                #= 9=# "Probability of being Impatient"     parameters.ν_Γ[1,1];
                #=10=# "Exogenous Risk-free Rate"           parameters.r_f;
                #=11=# "Multiplier of Incentive Constraint" parameters.λ;
                #=12=# "Additional Opportunity Cost"        (1.0+parameters.r_f)*parameters.λ;
                #=13=# "Leverage Ratio (Supply)"            parameters.ψ;
                #=14=# "Total Loans"                        variables.aggregate_var[1];
                #=15=# "Total Deposits"                     variables.aggregate_var[2];
                #=16=# "Net Worth"                          variables.aggregate_var[4];
                #=17=# "Leverage Ratio (Demand)"            variables.aggregate_var[5]]

hl_LR = Highlighter(f      = (data,i,j) -> i == 13 || i == 17,
                    crayon = Crayon(background = :light_blue))

pretty_table(data_spec, ["Name", "Value"];
             alignment=[:l,:r],
             formatters = ft_round(4),
             body_hlines = [7,9,13],
             highlighters = hl_LR)

# plot(parameters.a_grid_neg, variables.q, seriestype=:scatter)
# plot(parameters.a_grid_neg, -parameters.a_grid_neg.*variables.q, seriestype=:scatter)
# plot(parameters.a_grid_neg, variables.policy_d[1:parameters.a_size_neg,:,1])
# plot(parameters.a_grid_neg, variables.policy_a[1:parameters.a_size_neg,:,1])

#=
parameters = para_func()
para_targeted(x) = para_func(; λ = x)
solve_targeted(x) = solve_func!(var_func(para_targeted(x)), para_targeted(x))
λ_optimal = find_zero(solve_targeted, (0.021600396208, 0.021600396287), Bisection())
=#

#=================================#
# solve the model with MIT shocks #
#=================================#
parameters_MIT = para_func_MIT(parameters; ρ_z = 0.00, σ_z = 0.01, T_size = 80, time_varying_volatility = 0)
variables_MIT = var_func_MIT(λ_optimal, variables, parameters, parameters_MIT)
solve_func_MIT!(variables_MIT, parameters, parameters_MIT)

New_guess = Δ*variables_MIT.aggregate_var[6,:] + (1-Δ)*variables_MIT.λ_guess;
plot([variables_MIT.λ_guess, variables_MIT.aggregate_var[6,:], New_guess],
     label = ["Initial" "Simulated" "Updated"],
     legend = :bottomright,
     lw = 2)
