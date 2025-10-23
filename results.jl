using CairoMakie
using LaTeXStrings

pwd_ = pwd()
pwd_parts = split(pwd_, '/')
if pwd_parts[end] != "202510_NCKU"
    cd(pwd_ * "/results/figures/202510_NCKU/")
end

function save_fig(fig::Figure, filename::String, filetype::String)
    save(filename * "." * filetype, fig)
end

function plot_q_e1(v_old::MutableVariables, v_new::MutableVariables, p_old::NamedTuple; e1_i::Integer)
    colors = Makie.wong_colors()[1:p_old.e2_size]
    labels = [L"Low $e_2$", L"Mid $e_2$", L"High $e_2$"]
    fig = Figure(fontsize=32, size=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"a'", ylabel=L"q(a',\overline{e_1},e_2)")
    ylims!(ax, -0.05, 1.05)
    for e2_i in 1:p_old.e2_size
        lines!(ax, p_old.a_grid_neg, v_old.q[1:p_old.a_size_neg, e2_i, e1_i], color=colors[e2_i], label=labels[e2_i], linestyle=nothing, linewidth=4)
        lines!(ax, p_old.a_grid_neg, v_new.q[1:p_old.a_size_neg, e2_i, e1_i], color=colors[e2_i], linestyle=:dash, linewidth=4)
    end
    axislegend(ax; position=:lt, nbanks=1, patchsize=(40, 20))
    return fig
end

fig_q_low_e1 = plot_q_e1(variables_old, variables_new, parameters_old; e1_i=1);
save_fig(fig_q_low_e1, "fig_q_low_e1", "pdf");
fig_q_mid_e1 = plot_q_e1(variables_old, variables_new, parameters_old; e1_i=2);
save_fig(fig_q_mid_e1, "fig_q_mid_e1", "pdf");
fig_q_hig_e1 = plot_q_e1(variables_old, variables_new, parameters_old; e1_i=3);
save_fig(fig_q_hig_e1, "fig_q_hig_e1", "pdf");

function plot_q_e2(v_old::MutableVariables, v_new::MutableVariables, p_old::NamedTuple; e2_i::Integer)
    colors = Makie.wong_colors()[1:p_old.e1_size]
    labels = [L"Low $e_1$", L"Mid $e_1$", L"High $e_1$"]
    fig = Figure(fontsize=32, size=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"a'", ylabel=L"q(a',e_1,\overline{e_2})")
    ylims!(ax, -0.05, 1.05)
    for e1_i in 1:p_old.e1_size
        lines!(ax, p_old.a_grid_neg, v_old.q[1:p_old.a_size_neg, e2_i, e1_i], color=colors[e1_i], label=labels[e1_i], linestyle=nothing, linewidth=4)
        lines!(ax, p_old.a_grid_neg, v_new.q[1:p_old.a_size_neg, e2_i, e1_i], color=colors[e1_i], linestyle=:dash, linewidth=4)
    end
    axislegend(ax; position=:lt, nbanks=1, patchsize=(40, 20))
    return fig
end

fig_q_low_e2 = plot_q_e2(variables_old, variables_new, parameters_old; e2_i=1);
save_fig(fig_q_low_e2, "fig_q_low_e2", "pdf");
fig_q_mid_e2 = plot_q_e2(variables_old, variables_new, parameters_old; e2_i=2);
save_fig(fig_q_mid_e2, "fig_q_mid_e2", "pdf");
fig_q_hig_e2 = plot_q_e2(variables_old, variables_new, parameters_old; e2_i=3);
save_fig(fig_q_hig_e2, "fig_q_hig_e2", "pdf");

age_groups(n::Integer; start::Int=21, width::Int=10) = ["$(lo)-$(lo + width - 1)" for lo in start:width:start+width*(n-1)]

function plot_lifecycle(v_g_old::GroupVariables, v_g_new::GroupVariables; field_name::Symbol, ag_size::Integer)
    ag_size = ag_size ≤ length(v_g_old.x_dist) ? ag_size : length(v_g_old.x_dist)
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

fig_ls_d = plot_lifecycle(ag_mnts_old, ag_mnts_new; field_name=:share_of_filers_x, ag_size=6);
save_fig(fig_ls_d, "fig_ls_d", "pdf");
fig_ls_a_neg = plot_lifecycle(ag_mnts_old, ag_mnts_new; field_name=:share_in_debts_x, ag_size=6);
save_fig(fig_ls_a_neg, "fig_ls_a_neg", "pdf");
fig_ls_ir = plot_lifecycle(ag_mnts_old, ag_mnts_new; field_name=:avg_loan_rate_x, ag_size=6);
save_fig(fig_ls_ir, "fig_ls_ir", "pdf");

function newborn_welfare(v_old::MutableVariables, v_new::MutableVariables, p_old::NamedTuple)

    @unpack a_ind_zero, e1_size, e1_G, e2_size, e2_G, e3_size, e3_G, γ = p_old

    @views V_nb_old = v_old.V[a_ind_zero, :, :, :]
    @views V_nb_new = v_new.V[a_ind_zero, :, :, :]
    G_e123_Γ = reshape(e1_G, (1, 1, e1_size)) .* reshape(e2_G, (1, e2_size, 1)) .* reshape(e3_G, (e3_size, 1, 1))
    G_e23_Γ = reshape(e2_G, (1, e2_size)) .* reshape(e3_G, (e3_size, 1))

    V_nb_old_sum = sum(V_nb_old .* G_e123_Γ)
    V_nb_new_sum = sum(V_nb_new .* G_e123_Γ)
    welfare_CEV_newborn = 100 * ((V_nb_new_sum / V_nb_old_sum)^(1.0 / (1.0 - γ)) - 1.0)

    V_nb_old_sum_e1 = zeros(e1_size)
    V_nb_new_sum_e1 = zeros(e1_size)
    welfare_CEV_newborn_e1 = zeros(e1_size)
    for e1_i in 1:e1_size
        V_nb_old_sum_e1[e1_i] = sum(V_nb_old[:, :, e1_i] .* G_e23_Γ)
        V_nb_new_sum_e1[e1_i] = sum(V_nb_new[:, :, e1_i] .* G_e23_Γ)
        welfare_CEV_newborn_e1[e1_i] = 100 * ((V_nb_new_sum_e1[e1_i] / V_nb_old_sum_e1[e1_i])^(1.0 / (1.0 - γ)) - 1.0)
    end

    return welfare_CEV_newborn, welfare_CEV_newborn_e1
end

welfare_CEV_newborn, welfare_CEV_newborn_e1 = newborn_welfare(variables_old, variables_new, parameters_old)
welfare_CEV_newborn_NFFs, welfare_CEV_newborn_e1_NFFs = newborn_welfare(variables_old, variables_new_NFFs, parameters_old)