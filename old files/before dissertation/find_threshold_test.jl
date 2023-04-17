
a_p_i = 1
parameters.a_grid[a_p_i]

e_grid_itp = collect(-10:0.01:1)

V_diff = variables.V_nd[a_p_i,:,1] .- variables.V_d[:,1]
V_diff_itp = Akima(e_grid, V_diff)

plot(e_grid_itp, V_diff_itp.(e_grid_itp))
plot!(e_grid, V_diff, seriestype=:scatter, legend=:none)

V_nd_itp = Akima(e_grid, variables.V_nd[a_p_i,:,1])
V_d_itp = Akima(e_grid, variables.V_d[:,1])
V_diff_itp_1(e) = V_nd_itp(e) - V_d_itp(e)

plot(e_grid_itp, V_diff_itp_1.(e_grid_itp))
plot!(e_grid, V_diff, seriestype=:scatter, legend=:none)

plot(e_grid_itp, V_nd_itp.(e_grid_itp))
plot!(e_grid, variables.V_nd[a_p_i,:,1], seriestype=:scatter)

plot(e_grid_itp, V_d_itp.(e_grid_itp))
plot!(e_grid, variables.V_d[:,1], seriestype=:scatter)
