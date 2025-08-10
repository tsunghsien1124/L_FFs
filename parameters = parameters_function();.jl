parameters = parameters_function();
variables = variables_function(parameters, λ = 0.0);
V_p = rand(Float64, size(similar(variables.V)));
V_pos_p = rand(Float64, size(similar(variables.V_pos)));

function E_V_function_1!(
    V_p::Array{Float64,5},        # (a′, e3′, ν′, e2′, e1)
    V_pos_p::Array{Float64,5},    # (a′≥0, e3′, ν′, e2′, e1)
    variables::Mutable_Variables, # has EV::Array{Float64,4}, EV_pos::Array{Float64,4}
    parameters::NamedTuple
)
    """
    Fill variables.EV[a′, ν, e2, e1] and variables.EV_pos[a′≥0, ν, e2, e1]
    using precomputed Γ[e3′, ν′, e2′, ν, e2] and a batched loop over `loop_EV`.
    """
    @unpack e_3_size, ν_size, e_2_size, a_ind_zero, Γ, loop_EV = parameters
    @inline Nsum = e_3_size * ν_size * e_2_size

    @inbounds @batch for (e_1_i, e_2_i, ν_i, a_p_i) in loop_EV
        # read-only slices once per task
        Γ_block  = @view Γ[:, :, :, ν_i, e_2_i]               # (e3′, ν′, e2′)
        V_block  = @view V_p[a_p_i, :, :, :, e_1_i]            # (e3′, ν′, e2′)

        # zero-copy flatten: columns iterate e3′ fast, then ν′, then e2′
        g  = reshape(Γ_block,  Nsum)
        v  = reshape(V_block,  Nsum)

        if a_p_i > a_ind_zero
            a_pos_i = a_p_i - a_ind_zero + 1
            Vpos_block = @view V_pos_p[a_pos_i, :, :, :, e_1_i]
            vp = reshape(Vpos_block, Nsum)

            # fused pass: compute EV and EV_pos together
            variables.EV[a_p_i, ν_i, e_2_i, e_1_i] = 0.0
            variables.EV_pos[a_pos_i, ν_i, e_2_i, e_1_i] = 0.0
            @inbounds @simd for k in 1:Nsum
                variables.EV[a_p_i, ν_i, e_2_i, e_1_i]  += g[k] * v[k]
                variables.EV_pos[a_pos_i, ν_i, e_2_i, e_1_i] += g[k] * vp[k]
            end
        else
            # only EV (a′<0 has no EV_pos)
            variables.EV[a_p_i, ν_i, e_2_i, e_1_i] = 0.0
            @inbounds @simd for k in 1:Nsum
                variables.EV[a_p_i, ν_i, e_2_i, e_1_i] += g[k] * v[k]
            end
        end
    end

    return nothing
end


@btime E_V_function!($V_p, $V_pos_p, $variables, $parameters);
@btime E_V_function_1!($V_p, $V_pos_p, $variables, $parameters);
