function solve_ss()

    using Roots

    β = 0.96
    σ = 3
    λ_B = 0.8
    rf = 0.03
    rb = 0.04
    L = 10
    G_No_N = λ_B*((rb-rf)*L+(1+rf))
    G_Nn_ωN = (1-λ_B)*(1+rb)*L
    ω = (1-G_No_N) / G_Nn_ωN
    G_Nn_N = ω*G_Nn_ωN
    θ = ((β*(1-λ_B))/(λ_B*L))*(G_No_N/(1-β*G_No_N))
    ϕ = L*θ
    Λ = β*(1-λ_B+λ_B*ϕ)
    γ = (Λ*(rb-rf))/θ

    rb_MIT = (G_N_MIT-λ_B*(1+rf)*(1-L)) / ((λ_B+ω*(1-λ_B))*L)
end
