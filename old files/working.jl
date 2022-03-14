function solve_ss_func(parameters::NamedTuple)
    #-----------------------#
    # compute steady state. #
    #-----------------------#
    @unpack β, σ, λ_B, L, r_bf, r_f = parameters
    r_b = r_f + r_bf
    G_No_N = λ_B*((r_b-r_f)*L+(1+r_f))
    G_Nn_ωN = (1-λ_B)*(1+r_b)*L
    ω = (1-G_No_N) / G_Nn_ωN
    G_Nn_N = ω*G_Nn_ωN
    θ = ((β*(1-λ_B))/(λ_B*L))*(G_No_N/(1-β*G_No_N))
    ϕ = L*θ
    Λ = β*(1-λ_B+λ_B*ϕ)
    γ = (Λ*(r_b-r_f))/θ
    return ω, θ
end


rb_MIT = (G_N_MIT-λ_B*(1+rf)*(1-L)) / ((λ_B+ω*(1-λ_B))*L)
