parameters = parameters_function();
variables = variables_function(parameters);
V_p = rand(Float64, size(similar(variables.V)));
V_pos_p = rand(Float64, size(similar(variables.V_pos)));

@btime E_V_function!($V_p, $V_pos_p, $variables, $parameters);