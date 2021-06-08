using Measures
using Plots

#========================================#
# Bond price across persistent endowment #
#========================================#
plot_col = 1
plot_row = 1
plot_q_index = findall(-4.0 .<= parameters.a_grid .<= 0.0)
plot_q = plot(
    size = (plot_col * 800, plot_row * 500),
    box = :on,
    legend = :topleft,
    ylimit = [0.0, 1.0],
    yticks = 0.0:0.2:1.0,
    xtickfont = font(12, "Computer Modern", :black),
    ytickfont = font(12, "Computer Modern", :black),
    titlefont = font(18, "Computer Modern", :black),
    guidefont = font(16, "Computer Modern", :black),
    legendfont = font(14, "Computer Modern", :black),
    title = "\$ q(a',e) \$",
)
plot_q = plot!(
    parameters.a_grid[plot_q_index],
    variables.q[plot_q_index, 1],
    linecolor = :blue,
    linewidth = 3,
    label = "\$ \\textrm{Low Endowment } (e_p = $(round(parameters.e_grid[1],digits=1))) \$",
    margin = 4mm,
)
plot_q = plot!(
    parameters.a_grid[plot_q_index],
    variables.q[plot_q_index, 5],
    linecolor = :red,
    linewidth = 3,
    label = "\$ \\textrm{Median Endowment } (e_p = $(round(parameters.e_grid[5],digits=1))) \$",
    margin = 4mm,
)

plot_q = plot!(
    parameters.a_grid[plot_q_index],
    variables.q[plot_q_index, end],
    linecolor = :black,
    linewidth = 3,
    label = "\$ \\textrm{High Endowment } (e_p = $(round(parameters.e_grid[end],digits=1))) \$",
    margin = 4mm,
)
plot_q
savefig(plot_q, "figures/plot_q.pdf")

#===============================================================#
# Risky discounted borrowing amount across persistent endowment #
#===============================================================#
plot_col = 1
plot_row = 1
plot_qa_index = findall(-4.0 .<= parameters.a_grid .<= 0.0)
plot_qa = plot(
    size = (plot_col * 800, plot_row * 500),
    box = :on,
    legend = :none,
    ylimit = [-1.5, 0.0],
    yticks = -1.5:0.3:0.0,
    xtickfont = font(12, "Computer Modern", :black),
    ytickfont = font(12, "Computer Modern", :black),
    titlefont = font(18, "Computer Modern", :black),
    guidefont = font(16, "Computer Modern", :black),
    legendfont = font(12, "Computer Modern", :black),
    title = "\$ q(a',e) \\cdot a'\$",
)
plot_qa = plot!(
    parameters.a_grid[plot_qa_index],
    variables.q[plot_qa_index, 1] .* parameters.a_grid[plot_qa_index],
    linecolor = :blue,
    linewidth = 3,
    label = "\$ \\textrm{High Endowment } (e_p = $(round(parameters.e_grid[1],digits=1))) \$",
    margin = 4mm,
)
plot_qa = vline!([variables.rbl[1, 1]], linecolor = :blue, linewidth = 2, linestyle = :dot, label = "")
plot_qa = plot!(
    parameters.a_grid[plot_qa_index],
    variables.q[plot_qa_index, 5] .* parameters.a_grid[plot_qa_index],
    linecolor = :red,
    linewidth = 3,
    label = "\$ \\textrm{Median Endowment } (e_p = $(round(parameters.e_grid[5],digits=1))) \$",
    margin = 4mm,
)
plot_qa = vline!([variables.rbl[5, 1]], linecolor = :red, linewidth = 2, linestyle = :dot, label = "")
plot_qa = plot!(
    parameters.a_grid[plot_qa_index],
    variables.q[plot_qa_index, end] .* parameters.a_grid[plot_qa_index],
    linecolor = :black,
    linewidth = 3,
    label = "\$ \\textrm{Low Endowment } (e_p = $(round(parameters.e_grid[end],digits=1))) \$",
    margin = 4mm,
)
plot_qa = vline!([variables.rbl[end, 1]], linecolor = :black, linewidth = 2, linestyle = :dot, label = "")
plot_qa
savefig(plot_qa, "figures/plot_qa.pdf")


#=============#
# Default set #
#=============#
plot(parameters.a_grid_neg, parameters.w * exp.(variables.threshold_e[1:parameters.a_ind_zero, 2, 2] .+ parameters.t_grid[2]))
plot(variables.rbl[], parameters.e_grid)


parameters_NFF_η_20 = parameters_function(λ = 0.0, η = 0.20)
variables_NFF_η_20 = variables_function(parameters_NFF_η_20)
solve_economy_function!(variables_NFF_η_20, parameters_NFF_η_20)

parameters_NFF_η_30 = parameters_function(λ = 0.0, η = 0.30)
variables_NFF_η_30 = variables_function(parameters_NFF_η_30)
solve_economy_function!(variables_NFF_η_30, parameters_NFF_η_30)

parameters_NFF_η_40 = parameters_function(λ = 0.0, η = 0.40)
variables_NFF_η_40 = variables_function(parameters_NFF_η_40)
solve_economy_function!(variables_NFF_η_40, parameters_NFF_η_40)

parameters_NFF_η_50 = parameters_function(λ = 0.0, η = 0.50)
variables_NFF_η_50 = variables_function(parameters_NFF_η_50)
solve_economy_function!(variables_NFF_η_50, parameters_NFF_η_50)

parameters_NFF_η_60 = parameters_function(λ = 0.0, η = 0.60)
variables_NFF_η_60 = variables_function(parameters_NFF_η_60)
solve_economy_function!(variables_NFF_η_60, parameters_NFF_η_60)

results_filers_NFF = zeros(3, 5)

(variables_FF.aggregate_variables.share_of_filers - variables_NFF_η_355.aggregate_variables.share_of_filers) * 100 / variables_NFF_η_355.aggregate_variables.share_of_filers
(variables_FF.aggregate_variables.share_of_involuntary_filers - variables_NFF_η_355.aggregate_variables.share_of_involuntary_filers) * 100 / variables_FF.aggregate_variables.share_of_involuntary_filers
(
    (variables_FF.aggregate_variables.share_of_filers - variables_FF.aggregate_variables.share_of_involuntary_filers) -
    (variables_NFF_η_355.aggregate_variables.share_of_filers - variables_NFF_η_355.aggregate_variables.share_of_involuntary_filers)
) * 100 / (variables_NFF_η_355.aggregate_variables.share_of_filers - variables_NFF_η_355.aggregate_variables.share_of_involuntary_filers)


#================#
# Checking plots #
#================#
e_label = round.(exp.(parameters.e_grid), digits = 2)'
a_plot_index = findall(-3.0 .<= parameters.a_grid .<= 0.0)
plot(parameters.a_grid[a_plot_index], variables.q[a_plot_index, :], legend = :topleft, label = e_label)
# plot(parameters.a_grid, variables.q, legend = :topleft, label = e_label)

e_plot_i = 1
q_itp = Akima(parameters.a_grid, variables.q[:, e_plot_i])
a_grid_plot = findall(-0.5 .<= parameters.a_grid .<= 0.0)
plot(parameters.a_grid[a_grid_plot], q_itp.(parameters.a_grid[a_grid_plot]), legend = :topleft, label = "e = $(parameters.e_grid[e_plot_i])")
plot!(parameters.a_grid[a_grid_plot], variables.q[a_grid_plot, e_plot_i], seriestype = :scatter, label = "")

plot(parameters.a_grid_neg, variables.q[1:parameters.a_size_neg, :] .* parameters.a_grid_neg, legend = :left, label = e_label)
plot!(variables.rbl[:, 1], variables.rbl[:, 2], label = "rbl", seriestype = :scatter)
# plot!(parameters.a_grid_neg, parameters.a_grid_neg, lc = :black, label = "")

plot(parameters.a_grid, variables.q .* parameters.a_grid, legend = :bottomright, label = e_label)
plot!(variables.rbl[:, 1], variables.rbl[:, 2], label = "rbl", seriestype = :scatter)
plot!(parameters.a_grid, parameters.a_grid, lc = :black, label = "")

e_plot_i = 1
qa_itp = Akima(parameters.a_grid, variables.q[:, e_plot_i] .* parameters.a_grid)
a_grid_plot = findall(-2.0 .<= parameters.a_grid .<= 0.5)
plot(parameters.a_grid[a_grid_plot], qa_itp.(parameters.a_grid[a_grid_plot]), legend = :topleft, label = "e = $(parameters.e_grid[e_plot_i])")
plot!(parameters.a_grid[a_grid_plot], variables.q[a_grid_plot, e_plot_i] .* parameters.a_grid[a_grid_plot], seriestype = :scatter, label = "")
plot!(parameters.a_grid[a_grid_plot], parameters.a_grid[a_grid_plot], lc = :black, label = "")
hline!([0.0], lc = :black, label = "")
vline!([0.0], lc = :black, label = "")


t_plot_i = 1
ν_plot_i = 2
plot(parameters.a_grid_neg, variables.V[1:parameters.a_ind_zero, :, t_plot_i, ν_plot_i], legend = :bottomleft, label = e_label)
plot!(variables.threshold_a[:, t_plot_i, ν_plot_i], variables.V_d[:, t_plot_i, ν_plot_i], label = "defaulting debt level", seriestype = :scatter)
hline!([0.0], lc = :black, label = "")
vline!([0.0], lc = :black, label = "")

plot(parameters.a_grid, variables.V[:, :, 2, 2], legend = :bottomleft, label = e_label)
plot!(variables.threshold_a[:, 2, 2], variables.V_d[:, 2, 2], label = "defaulting debt level", seriestype = :scatter)
hline!([0.0], lc = :black, label = "")
vline!([0.0], lc = :black, label = "")

any(variables.V .< 0.0)

plot(-variables.threshold_a[:, 1], parameters.w * exp.(parameters.e_grid), legend = :none, markershape = :circle, xlabel = "defaulting debt level", ylabel = "w*exp(e)")

plot(parameters.a_grid_neg, variables.threshold_e[1:parameters.a_ind_zero, 1], legend = :none, xlabel = "debt level", ylabel = "defaulting e level")
plot!(variables.threshold_a[:, 1], parameters.e_grid, seriestype = :scatter)

plot(parameters.a_grid_neg, parameters.w .* exp.(variables.threshold_e[1:parameters.a_ind_zero, 1]), legend = :none, xlabel = "debt level", ylabel = "defaulting w*exp(e) level")
plot!(variables.threshold_a[:, 1], parameters.w * exp.(parameters.e_grid), seriestype = :scatter)

plot(parameters.a_grid, parameters.w .* exp.(variables.threshold_e[:, 1]), legend = :none, xlabel = "debt level", ylabel = "defaulting w*exp(e) level")
plot!(variables.threshold_a[:, 1], parameters.w * exp.(parameters.e_grid), seriestype = :scatter)



η_grid = results_A_NFF[1, :]
η_size = length(η_grid)
CEV_comparison_NFF = zeros(η_size)
CEV_comparison_FF = zeros(η_size)
for η_i = 1:η_size
    @inbounds CEV_comparison_NFF[η_i] = sum(results_CEV_NFF[:, :, :, :, η_i] .* results_μ_NFF[:, :, :, :, η_i])
    @inbounds CEV_comparison_FF[η_i] = sum(results_CEV_FF[:, :, :, :, η_i] .* results_μ_FF[:, :, :, :, η_i])
end

plot(η_grid, CEV_comparison_NFF)
plot!(η_grid, CEV_comparison_FF)

plot_col = 1
plot_row = 1

plot_all = plot(
    size = (plot_col * 600, plot_row * 500),
    box = :on,
    legend = :topleft,
    xtickfont = font(12, "Computer Modern", :black),
    ytickfont = font(12, "Computer Modern", :black),
    titlefont = font(18, "Computer Modern", :black),
    guidefont = font(16, "Computer Modern", :black),
    legendfont = font(12, "Computer Modern", :black),
)
plot_all = plot!(
    η_grid,
    CEV_comparison_NFF,
    xticks = 0.2:0.1:0.8,
    seriestype = :path,
    markershapes = :circle,
    markercolor = :auto,
    markersize = 5,
    markerstrokecolor = :auto,
    lw = 3,
    label = "Without Financial Frictions",
    margin = 4mm,
)
plot_all = vline!([η_grid[argmax(CEV_comparison_NFF)]], color = 1, lw = 3, ls = :dot, label = "")

plot_all = plot!(
    η_grid,
    CEV_comparison_FF,
    seriestype = :path,
    markershapes = :square,
    markercolor = :auto,
    markersize = 5,
    markerstrokecolor = :auto,
    lw = 3,
    color = 2,
    label = "With Financial Frictions",
    xlabel = "\$ \\textrm{wage garnishment rate } (\\eta) \$",
    title = "Welfare (CEV)",
    margin = 4mm,
)

plot_all = vline!([η_grid[argmax(CEV_comparison_FF)]], color = 2, lw = 3, ls = :dot, label = "")
