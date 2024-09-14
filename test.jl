using BenchmarkTools

function EV_function_1!(E_V::Array{Float64,3}, V_p::Array{Float64,5}, parameters::NamedTuple)
  """
  construct expected value functions
  """

  # unpack parameters
  @unpack e_1_size, e_2_size, e_2_Γ, e_3_size, e_3_Γ, ν_size, ν_Γ, a_size, a_size_pos, ρ, β = parameters

  # reshape all objects
  e_1_Γ = reshape(Matrix{Float64}(I, e_1_size, e_1_size), (1, e_1_size, 1, 1, 1, e_1_size, 1))
  e_2_Γ = reshape(transpose(e_2_Γ), (1, 1, e_2_size, 1, 1, 1, e_2_size))
  e_3_Γ = reshape(e_3_Γ, (1, 1, 1, e_3_size, 1, 1, 1))
  ν_Γ = reshape(ν_Γ, (1, 1, 1, 1, ν_size, 1, 1))
  V_p_res = reshape(V_p, (a_size, e_1_size, e_2_size, e_3_size, ν_size, 1, 1))

  # update expected value
  E_V .= ρ .* β .* dropdims(sum(e_1_Γ .* e_2_Γ .* e_3_Γ .* ν_Γ .* V_p_res, dims=(2, 3, 4, 5)), dims=(2, 3, 4, 5))

  # replace NaN with -Inf
  replace!(E_V, NaN => -Inf)

  # return results
  return nothing
end

function EV_function_2!(E_V::Array{Float64,3}, V_p::Array{Float64,5}, parameters::NamedTuple)
  """
  construct expected value functions
  """

  # unpack parameters
  @unpack e_1_size, e_2_size, e_2_Γ, e_3_size, e_3_Γ, ν_size, ν_Γ, a_size, a_size_pos, ρ, β = parameters

  # reshape all objects
  for e_2_i = 1:e_2_size, e_1_i = 1:e_1_size, a_p_i = 1:a_size
    E_V[a_p_i, e_1_i, e_2_i] = 0.0
    for ν_p_i = 1:ν_size, e_3_p_i = 1:e_3_size, e_2_p_i = 1:e_2_size
      E_V[a_p_i, e_1_i, e_2_i] += ρ * β * e_2_Γ[e_2_i, e_2_p_i] * e_3_Γ[e_3_p_i] * ν_Γ[ν_p_i] * V_p[a_p_i, e_1_i, e_2_p_i, e_3_p_i, ν_p_i]
    end
  end 

  # replace NaN with -Inf
  replace!(E_V, NaN => -Inf)

  # return results
  return nothing
end

function EV_function_3!(E_V::Array{Float64,3}, parameters::NamedTuple, e_1_Γ, e_2_Γ, e_3_Γ, ν_Γ, V_p_res)
  """
  construct expected value functions
  """

  # unpack parameters
  @unpack ρ, β = parameters

  # update expected value
  E_V .= ρ .* β .* dropdims(sum(e_1_Γ .* e_2_Γ .* e_3_Γ .* ν_Γ .* V_p_res, dims=(2, 3, 4, 5)), dims=(2, 3, 4, 5))

  # replace NaN with -Inf
  replace!(E_V, NaN => -Inf)

  # return results
  return nothing
end

V_p = rand(parameters.a_size, parameters.e_1_size, parameters.e_2_size, parameters.e_3_size, parameters.ν_size);
E_V_1 = zeros(parameters.a_size, parameters.e_1_size, parameters.e_2_size);
E_V_2 = zeros(parameters.a_size, parameters.e_1_size, parameters.e_2_size);
E_V_3 = zeros(parameters.a_size, parameters.e_1_size, parameters.e_2_size);

# reshape all objects
@unpack e_1_size, e_2_size, e_2_Γ, e_3_size, e_3_Γ, ν_size, ν_Γ, a_size, a_size_pos = parameters
e_1_Γ = reshape(Matrix{Float64}(I, e_1_size, e_1_size), (1, e_1_size, 1, 1, 1, e_1_size, 1))
e_2_Γ = reshape(transpose(e_2_Γ), (1, 1, e_2_size, 1, 1, 1, e_2_size))
e_3_Γ = reshape(e_3_Γ, (1, 1, 1, e_3_size, 1, 1, 1))
ν_Γ = reshape(ν_Γ, (1, 1, 1, 1, ν_size, 1, 1))
V_p_res = reshape(V_p, (a_size, e_1_size, e_2_size, e_3_size, ν_size, 1, 1))

@btime EV_function_1!($E_V_1, $V_p, $parameters);
@btime EV_function_2!($E_V_2, $V_p, $parameters);
@btime EV_function_3!($E_V_3, $parameters, $e_1_Γ, $e_2_Γ, $e_3_Γ, $ν_Γ, $V_p_res);
@assert E_V_1 ≈ E_V_2 ≈ E_V_3