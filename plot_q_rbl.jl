using BSON: @save, @load
using Plots
using PlotThemes
using LaTeXStrings
using QuantEcon: tauchen

@load "Cont_results" q_cont a_grid_neg_cont a_ind_zero_cont
@load "Cont_results_eg" q_cont_eg a_grid_neg_cont_eg a_ind_zero_cont_eg
@load "DSS_results" q_dss a_grid_neg_dss a_ind_zero_dss
@load "DSS_results_eg" q_dss_eg a_grid_neg_dss_eg a_ind_zero_dss_eg

e_ρ = 0.95
e_σ = 0.10
e_size = 15
e_M = tauchen(e_size, e_ρ, e_σ, 0.0, 8)
e_Γ = e_M.p
e_grid = round.(exp.(collect(e_M.state_values)), digits = 4)

label_latex = reshape(latexstring.("\$",["e = $(e_grid[i])" for i in 1:e_size],"\$"),1,:)

plot(a_grid_neg_cont, q_cont[1:a_ind_zero_cont,:,1],
     linewidth = 1.5,
     xlabel = "\$a'\$", ylabel = "\$q(a',s)\$",
     xlims = (-4.0,0.0), ylims = (0.0,1.0),
     xticks = -4.0:0.5:0.0, yticks = 0.0:0.2:1.0,
     label = label_latex,
     legend = :none, legendfont = font(9),
     theme = theme(:default))

plot!(a_grid_neg_dss, q_dss[1:a_ind_zero_dss,1:e_size],
      lw = 1.5,
      color = collect(1:e_size)',
      linestyle = :dot)

savefig("plot_q.pdf")

plot(a_grid_neg_cont, repeat(a_grid_neg_cont,1,e_size).*q_cont[1:a_ind_zero_cont,:,1],
     linewidth = 1.5,
     xlabel = "\$a'\$", ylabel = "\$q(a',s)a'\$",
     xlims = (-4.0,0.0), ylims = (-3.0,0.0),
     xticks = -4.0:0.5:0.0, yticks = -3.0:0.5:0.0,
     label = label_latex,
     legend = :none, legendfont = font(9),
     theme = theme(:default))

plot!(a_grid_neg_dss, repeat(a_grid_neg_dss,1,e_size).*q_dss[1:a_ind_zero_dss,1:e_size],
      lw = 1.5,
      color = collect(1:e_size)',
      linestyle = :dot)

savefig("plot_rbl.pdf")

plot(a_grid_neg_cont_eg, q_cont_eg[1:a_ind_zero_cont_eg,:,1],
     linewidth = 1.5,
     xlabel = "\$a'\$", ylabel = "\$q(a',s)\$",
     xlims = (-4.5,0.0), ylims = (0.0,1.0),
     xticks = -4.5:0.5:0.0, yticks = 0.0:0.2:1.0,
     label = label_latex,
     legend = :none, legendfont = font(9),
     theme = theme(:default))

plot!(a_grid_neg_dss_eg, q_dss_eg[1:a_ind_zero_dss_eg,1:e_size],
      lw = 1.5,
      color = collect(1:e_size)',
      linestyle = :dot)

savefig("plot_q_eg.pdf")

plot(a_grid_neg_cont_eg, repeat(a_grid_neg_cont_eg,1,e_size).*q_cont_eg[1:a_ind_zero_cont,:,1],
     linewidth = 1.5,
     xlabel = "\$a'\$", ylabel = "\$q(a',s)a'\$",
     xlims = (-4.5,0.0), ylims = (-4.0,0.0),
     xticks = -4.5:0.5:0.0, yticks = -3.5:0.5:0.0,
     label = label_latex,
     legend = :none, legendfont = font(9),
     theme = theme(:default))

plot!(a_grid_neg_dss_eg, repeat(a_grid_neg_dss_eg,1,e_size).*q_dss_eg[1:a_ind_zero_dss_eg,1:e_size],
      lw = 1.5,
      color = collect(1:e_size)',
      linestyle = :dot)

savefig("plot_rbl_eg.pdf")

#=
scatter!(a_grid_neg_dss, q_dss[1:a_ind_zero_dss,1:e_size],
         markershape = :hexagon,
         markersize = 1.5,
         markeralpha = 0.6,
         markercolor = collect(1:e_size)',
         markerstrokewidth = 0,
         markerstrokealpha = 0.2,
         markerstrokecolor = :black,
         markerstrokestyle = :dot)
