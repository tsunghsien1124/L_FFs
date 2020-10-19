using QuantEcon: rouwenhorst, tauchen, gridmake, MarkovChain, stationary_distributions

n = 15
ρ = 0.90
σ_1 = 0.10
σ_2 = 0.20


function tauchen_test(n, ρ, σ, μ)
    M = tauchen(n, ρ, σ, μ, 3)
    x = collect(M.state_values)
    ex = exp.(x)
    SD = stationary_distributions(M)[]
    X = sum(x .* SD)
    EX = sum(ex .* SD)
    return X, EX
end

X_1, EX_1 = tauchen_test(n, ρ, σ_1, 0)
X_1_adj, EX_1_adj = tauchen_test(n, ρ, σ_1, -σ_1^2/2)

[abs(EX_1 - 1), abs(EX_1_adj - 1)]

X_2, EX_2 = tauchen_test(n, ρ, σ_2, 0)
X_2_adj, EX_2_adj = tauchen_test(n, ρ, σ_2, -σ_2^2/2)

[abs(EX_2 - 1), abs(EX_2_adj - 1)]
