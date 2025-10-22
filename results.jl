using CairoMakie
using LaTeXStrings

function plot_q(v_old::MutableVariables, p_old::NamedTuple, v_new::MutableVariables, p_new::NamedTuple; e1_i::Integer)
    color_arrary = [:blue, :red, :black]
    # label_arrary = [L"Low $e_2", L"Mid $e_2", L"High $e_2" ]
    fig = Figure(fontsize=32, size=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"a'", ylabel=L"q(a',e_1,e_2)")
    ylims!(ax, -0.05, 1.05)
    for e2_i in 1:p_old.e2_size
        # label=label_arrary[e2_i]
        lines!(ax, p_old.a_grid_neg, v_old.q[1:p_old.a_size_neg, e2_i, e1_i], color=color_arrary[e2_i], linestyle=nothing, linewidth=4)
        lines!(ax, p_new.a_grid_neg, v_new.q[1:p_new.a_size_neg, e2_i, e1_i], color=color_arrary[e2_i], linestyle=:dash, linewidth=4)
    end
    # axislegend(position=:lt, nbanks=3, patchsize=(40, 20))
    return fig
end

fig_q_e1_1 = plot_q(variables_old, parameters_old, variables_new, parameters_new; e1_i = 1)
fig_q_e1_2 = plot_q(variables_old, parameters_old, variables_new, parameters_new; e1_i = 2)
fig_q_e1_3 = plot_q(variables_old, parameters_old, variables_new, parameters_new; e1_i = 3)

# save(PATH_FIG_para_x * FL * filename_x * ".pdf", fig)
# save(PATH_FIG_para_x * FL * filename_x * ".png", fig)

function plot_lifecycle(v_ls_old::LifecycleVariables, v_ls_new::LifecycleVariables; field_name::Symbol, ag_size::Integer)

    x = 1:ag_size
    y_old = getfield(v_ls_old, field_name)[1:ag_size]
    y_new = getfield(v_ls_new, field_name)[1:ag_size]
    fig = Figure(fontsize=32, size=(800, 600))
    ax = Axis(fig[1, 1])
    scatterlines!(ax, x, y_old, color=:blue, linestyle=nothing, linewidth=4, markersize = 20)
    scatterlines!(ax, x, y_new, color=:red, linestyle=:dash, linewidth=4, markersize = 20)
    # categorical_labels = ["Apple", "Banana", "Cherry", "A", "B"]
    # xticks!(ax, xtickrange=1:5, xticklabels=categorical_labels)
    fig
end

fig_ls_d = plot_lifecycle(variables_ls_old, variables_ls_new; field_name = :share_of_filers_ag, ag_size = 8)
fig_ls_a_neg = plot_lifecycle(variables_ls_old, variables_ls_new; field_name = :share_in_debts_ag, ag_size = 8)
fig_ls_ir = plot_lifecycle(variables_ls_old, variables_ls_new; field_name = :avg_loan_rate_ag, ag_size = 8)
