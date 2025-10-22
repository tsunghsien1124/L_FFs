using CairoMakie
using LaTeXStrings

function plot_q_e1(v_old::MutableVariables, p_old::NamedTuple, v_new::MutableVariables, p_new::NamedTuple; e1_i::Integer)
    colors = Makie.wong_colors()[1:p_old.e2_size]
    labels = [L"Low $e_2$", L"Mid $e_2$", L"High $e_2$"]
    fig = Figure(fontsize=32, size=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"a'", ylabel=L"q(a',e_1,e_2)")
    ylims!(ax, -0.05, 1.05)
    for e2_i in 1:p_old.e2_size
        lines!(ax, p_old.a_grid_neg, v_old.q[1:p_old.a_size_neg, e2_i, e1_i], color=colors[e2_i], label=labels[e2_i], linestyle=nothing, linewidth=4)
        lines!(ax, p_new.a_grid_neg, v_new.q[1:p_new.a_size_neg, e2_i, e1_i], color=colors[e2_i], linestyle=:dash, linewidth=4)
    end
    axislegend(ax; position=:lt, nbanks=1, patchsize=(40, 20))
    return fig
end

fig_q_low_e1 = plot_q_e1(variables_old, parameters_old, variables_new, parameters_new; e1_i=1)
fig_q_mid_e1 = plot_q_e1(variables_old, parameters_old, variables_new, parameters_new; e1_i=2)
fig_q_hig_e1 = plot_q_e1(variables_old, parameters_old, variables_new, parameters_new; e1_i=3)

function plot_q_e2(v_old::MutableVariables, p_old::NamedTuple, v_new::MutableVariables, p_new::NamedTuple; e2_i::Integer)
    colors = Makie.wong_colors()[1:p_old.e1_size]
    labels = [L"Low $e_1$", L"Mid $e_1$", L"High $e_1$"]
    fig = Figure(fontsize=32, size=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"a'", ylabel=L"q(a',e_1,e_2)")
    ylims!(ax, -0.05, 1.05)
    for e1_i in 1:p_old.e1_size
        lines!(ax, p_old.a_grid_neg, v_old.q[1:p_old.a_size_neg, e2_i, e1_i], color=colors[e1_i], label=labels[e1_i], linestyle=nothing, linewidth=4)
        lines!(ax, p_new.a_grid_neg, v_new.q[1:p_new.a_size_neg, e2_i, e1_i], color=colors[e1_i], linestyle=:dash, linewidth=4)
    end
    axislegend(ax; position=:lt, nbanks=1, patchsize=(40, 20))
    return fig
end

fig_q_low_e2 = plot_q_e2(variables_old, parameters_old, variables_new, parameters_new; e2_i=1)
fig_q_mid_e2 = plot_q_e2(variables_old, parameters_old, variables_new, parameters_new; e2_i=2)
fig_q_hig_e2 = plot_q_e2(variables_old, parameters_old, variables_new, parameters_new; e2_i=3)

# save(PATH_FIG_para_x * FL * filename_x * ".pdf", fig)
# save(PATH_FIG_para_x * FL * filename_x * ".png", fig)

age_groups(n::Integer; start::Int=21, width::Int=10) = ["$(lo)-$(lo + width - 1)" for lo in start:width:start+width*(n-1)]

function plot_lifecycle(v_g_old::GroupVariables, v_g_new::GroupVariables; field_name::Symbol, ag_size::Integer)
    ag_grp = age_groups(ag_size)
    y_old = getfield(v_g_old, field_name)[1:ag_size]
    y_new = getfield(v_g_new, field_name)[1:ag_size]
    colors = Makie.wong_colors()[1:2]
    labels = ["Benchmark", "BAPCPA"]
    fig = Figure(fontsize=32, size=(800, 600))
    ax = Axis(fig[1, 1], xticks=(1:ag_size, ag_grp))
    x = 1:ag_size
    G = 2                                               # number of groups (old/new)
    cluster = 0.8                                       # total width allocated per category on the x-axis
    w = cluster / G                                     # individual bar width
    offs = (-cluster / 2) .+ (0:G-1) .* w .+ (w / 2)    # symmetric offsets
    barplot!(ax, x .+ offs[1], y_old; width=w, color=colors[1], label=labels[1])
    barplot!(ax, x .+ offs[2], y_new; width=w, color=colors[2], label=labels[2])
    axislegend(ax; position=:rt, nbanks=1, patchsize=(40, 20))
    return fig
end

fig_ls_d = plot_lifecycle(variables_g_old, variables_g_new; field_name=:share_of_filers_ag, ag_size=4)
fig_ls_a_neg = plot_lifecycle(variables_g_old, variables_g_new; field_name=:share_in_debts_ag, ag_size=4)
fig_ls_ir = plot_lifecycle(variables_g_old, variables_g_new; field_name=:avg_loan_rate_ag, ag_size=4)