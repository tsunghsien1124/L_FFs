
parameters_ref = para_func(; λ = 0.0)

@unpack α, K2Y, δ, β_B, θ, ψ, e_SS = parameters_ref

β_B = 0.96
ψ = 0.95
θ = 0.40
δ = 0.10
α = 0.33

i = (α/K2Y) - δ
λ_min = 0.0
λ_max = 1.0 - (β_B*ψ*(1+i))^(1/2)
λ_grid = collect(λ_min:0.001:λ_max)
λ_size = length(λ_grid)

γ_grid = zeros(λ_size)
Λ_grid = zeros(λ_size)
LR_grid = zeros(λ_size)
AD_grid = zeros(λ_size)
r_lp_grid = zeros(λ_size)
r_k_grid = zeros(λ_size)
K_grid = zeros(λ_size)
w_grid = zeros(λ_size)

for λ_i in 1:λ_size
    λ = λ_grid[λ_i]
    γ_grid[λ_i] = (β_B*(1.0-ψ)*(1.0+i)) / ((1.0-λ)-β_B*ψ*(1.0+i))
    Λ_grid[λ_i] = β_B*(1.0-ψ+ψ*γ_grid[λ_i])
    LR_grid[λ_i] = γ_grid[λ_i]/θ
    AD_grid[λ_i] = LR_grid[λ_i]/(LR_grid[λ_i]-1.0)
    r_lp_grid[λ_i] = λ*θ/Λ_grid[λ_i]
    r_k_grid[λ_i] = i + r_lp_grid[λ_i]
    K_grid[λ_i] = exp(e_SS)*(α/(r_k_grid[λ_i]+δ))^(1.0/(1.0-α))
    w_grid[λ_i] = (1.0-α)*(K_grid[λ_i]^α)*(exp(e_SS)^(-α))
end

plot(r_lp_grid, AD_grid, seriestype=:scatter)
plot!(r_lp_grid, LR_grid, seriestype=:scatter)
