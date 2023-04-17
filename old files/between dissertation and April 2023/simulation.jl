function simulation(variables::Mutable_Variables, parameters::NamedTuple; num_hh::Integer = 20000, num_periods::Integer = 2000, burn_in::Integer = 100)

    # initialization
    num_periods = num_periods + 1
    Random.seed!(1124)

    # endogenous state or choice variables
    panel_asset = zeros(Int, num_hh, num_periods)
    panel_history = zeros(Int, num_hh, num_periods)
    panel_default = zeros(Int, num_hh, num_periods)
    panel_age = zeros(Int, num_hh, num_periods)
    panel_consumption = zeros(num_hh, num_periods)

    # exogenous variables
    shock_ρ = rand(Categorical([parameters.ρ, 1-parameters.ρ]), (num_hh, num_periods))
    shock_e_1 = zeros(Int, num_hh, num_periods)
    shock_e_2 = zeros(Int, num_hh, num_periods)
    shock_e_3 = zeros(Int, num_hh, num_periods)
    shock_ν = zeros(Int, num_hh, num_periods)

    # Loop over HHs and Time periods
    @showprogress 1 "Computing..." for period_i in 1:(num_periods-1)
        Threads.@threads for hh_i in 1:num_hh
            if period_i == 1 || shock_ρ[hh_i,period_i] == 2

                # initiate states for newborns
                panel_age[hh_i,period_i] = 1
                e_1_i = rand(Categorical(vec(parameters.G_e_1)))
                shock_e_1[hh_i,period_i] = e_1_i
                e_2_i = rand(Categorical(vec(parameters.G_e_2)))
                shock_e_2[hh_i,period_i] = e_2_i
                e_3_i = rand(Categorical(vec(parameters.G_e_3)))
                shock_e_3[hh_i,period_i] = e_3_i
                ν_i = rand(Categorical(vec(parameters.G_ν)))
                shock_ν[hh_i,period_i] = ν_i
                earnings = variables.aggregate_prices.w_λ * exp(parameters.e_1_grid[e_1_i] + parameters.e_2_grid[e_2_i] + parameters.e_3_grid[e_3_i])
                asset_i = parameters.a_ind_zero
                panel_asset[hh_i,period_i] = asset_i

                # compute choices
                default_prob = variables.policy_d[asset_i,e_1_i,e_2_i,e_3_i,ν_i]
                default_i = rand(Categorical(vec([default_prob,1.0-default_prob])))
                if default_i == 1
                    panel_asset[hh_i,period_i+1] = parameters.a_ind_zero
                    panel_default[hh_i,period_i] = default_i
                    panel_history[hh_i,period_i] = 1
                    panel_consumption[hh_i,period_i] = (1-parameters.η)*earnings
                else
                    asset_p = variables.policy_a[asset_i,e_1_i,e_2_i,e_3_i,ν_i]
                    asset_p_lb_i = findall(parameters.a_grid .<= asset_p)[end]
                    asset_p_ub_i = findall(asset_p .<= parameters.a_grid)[1]
                    if asset_p_lb_i != asset_p_ub_i
                        @inbounds asset_p_lower = parameters.a_grid[asset_p_lb_i]
                        @inbounds asset_p_upper = parameters.a_grid[asset_p_ub_i]
                        weight_lower = (asset_p_upper - asset_p) / (asset_p_upper - asset_p_lower)
                        weight_upper = (asset_p - asset_p_lower) / (asset_p_upper - asset_p_lower)
                        asset_p_i = rand(Categorical(vec([weight_lower,weight_upper])))
                        if asset_p_i == 1
                            asset_p_i = asset_p_lb_i
                        else
                            asset_p_i = asset_p_ub_i
                        end
                    else
                        asset_p_i = asset_p_ub_i
                    end
                    panel_asset[hh_i,period_i+1] = asset_p_i
                    panel_consumption[hh_i,period_i] = earnings - variables.q[asset_p_i,e_1_i,e_2_i] * parameters.a_grid[asset_p_i]
                end

            else

                # extract states
                panel_age[hh_i,period_i] = panel_age[hh_i,period_i-1] + 1
                e_1_i = shock_e_1[hh_i,period_i-1]
                shock_e_1[hh_i,period_i] = e_1_i
                e_2_i = rand(Categorical(parameters.e_2_Γ[shock_e_2[hh_i,period_i-1],:]))
                shock_e_2[hh_i,period_i] = e_2_i
                e_3_i = rand(Categorical(parameters.e_3_Γ))
                shock_e_3[hh_i,period_i] = e_3_i
                ν_i = rand(Categorical(vec(parameters.ν_Γ)))
                shock_ν[hh_i,period_i] = ν_i
                earnings = variables.aggregate_prices.w_λ * exp(parameters.e_1_grid[e_1_i] + parameters.e_2_grid[e_2_i] + parameters.e_3_grid[e_3_i])
                asset_i = panel_asset[hh_i,period_i]
                asset = parameters.a_grid[asset_i]

                if panel_history[hh_i,period_i-1] == 1

                    history_i = rand(Categorical(vec([1.0-parameters.p_h,parameters.p_h])))

                    if history_i == 1
                        panel_history[hh_i,period_i] = history_i
                        asset_i = asset_i - parameters.a_ind_zero + 1
                        asset_p = variables.policy_pos_a[asset_i,e_1_i,e_2_i,e_3_i,ν_i]
                        asset_p_lb_i = findall(parameters.a_grid .<= asset_p)[end]
                        asset_p_ub_i = findall(asset_p .<= parameters.a_grid)[1]
                        if asset_p_lb_i != asset_p_ub_i
                            @inbounds asset_p_lower = parameters.a_grid[asset_p_lb_i]
                            @inbounds asset_p_upper = parameters.a_grid[asset_p_ub_i]
                            weight_lower = (asset_p_upper - asset_p) / (asset_p_upper - asset_p_lower)
                            weight_upper = (asset_p - asset_p_lower) / (asset_p_upper - asset_p_lower)
                            asset_p_i = rand(Categorical(vec([weight_lower,weight_upper])))
                            if asset_p_i == 1
                                asset_p_i = asset_p_lb_i
                            else
                                asset_p_i = asset_p_ub_i
                            end
                        else
                            asset_p_i = asset_p_ub_i
                        end
                        panel_asset[hh_i,period_i+1] = asset_p_i
                        panel_consumption[hh_i,period_i] = earnings + asset - variables.q[asset_p_i,e_1_i,e_2_i] * parameters.a_grid[asset_p_i]

                    else

                        default_prob = variables.policy_d[asset_i,e_1_i,e_2_i,e_3_i,ν_i]
                        default_i = rand(Categorical(vec([default_prob,1.0-default_prob])))
                        if default_i == 1
                            panel_asset[hh_i,period_i+1] = parameters.a_ind_zero
                            panel_default[hh_i,period_i] = default_i
                            panel_history[hh_i,period_i] = 1
                            panel_consumption[hh_i,period_i] = (1-parameters.η)*earnings
                        else
                            asset_p = variables.policy_a[asset_i,e_1_i,e_2_i,e_3_i,ν_i]
                            asset_p_lb_i = findall(parameters.a_grid .<= asset_p)[end]
                            asset_p_ub_i = findall(asset_p .<= parameters.a_grid)[1]
                            if asset_p_lb_i != asset_p_ub_i
                                @inbounds asset_p_lower = parameters.a_grid[asset_p_lb_i]
                                @inbounds asset_p_upper = parameters.a_grid[asset_p_ub_i]
                                weight_lower = (asset_p_upper - asset_p) / (asset_p_upper - asset_p_lower)
                                weight_upper = (asset_p - asset_p_lower) / (asset_p_upper - asset_p_lower)
                                asset_p_i = rand(Categorical(vec([weight_lower,weight_upper])))
                                if asset_p_i == 1
                                    asset_p_i = asset_p_lb_i
                                else
                                    asset_p_i = asset_p_ub_i
                                end
                            else
                                asset_p_i = asset_p_ub_i
                            end
                            panel_asset[hh_i,period_i+1] = asset_p_i
                            panel_consumption[hh_i,period_i] = earnings + asset - variables.q[asset_p_i,e_1_i,e_2_i] * parameters.a_grid[asset_p_i]
                        end
                    end

                else

                    default_prob = variables.policy_d[asset_i,e_1_i,e_2_i,e_3_i,ν_i]
                    default_i = rand(Categorical(vec([default_prob,1.0-default_prob])))
                    if default_i == 1
                        panel_asset[hh_i,period_i+1] = parameters.a_ind_zero
                        panel_default[hh_i,period_i] = default_i
                        panel_history[hh_i,period_i] = 1
                        panel_consumption[hh_i,period_i] = (1-parameters.η)*earnings
                    else
                        asset_p = variables.policy_a[asset_i,e_1_i,e_2_i,e_3_i,ν_i]
                        asset_p_lb_i = findall(parameters.a_grid .<= asset_p)[end]
                        asset_p_ub_i = findall(asset_p .<= parameters.a_grid)[1]
                        if asset_p_lb_i != asset_p_ub_i
                            @inbounds asset_p_lower = parameters.a_grid[asset_p_lb_i]
                            @inbounds asset_p_upper = parameters.a_grid[asset_p_ub_i]
                            weight_lower = (asset_p_upper - asset_p) / (asset_p_upper - asset_p_lower)
                            weight_upper = (asset_p - asset_p_lower) / (asset_p_upper - asset_p_lower)
                            asset_p_i = rand(Categorical(vec([weight_lower,weight_upper])))
                            if asset_p_i == 1
                                asset_p_i = asset_p_lb_i
                            else
                                asset_p_i = asset_p_ub_i
                            end
                        else
                            asset_p_i = asset_p_ub_i
                        end
                        panel_asset[hh_i,period_i+1] = asset_p_i
                        panel_consumption[hh_i,period_i] = earnings + asset - variables.q[asset_p_i,e_1_i,e_2_i] * parameters.a_grid[asset_p_i]
                    end
                end
            end
        end
        # println("Computing the period $period_i")
    end

    # Cut burn-in and last period
    panel_asset = panel_asset[:,burn_in+1:end-1]
    panel_history = panel_history[:,burn_in+1:end-1]
    panel_default = panel_default[:,burn_in+1:end-1]
    panel_age = panel_age[:,burn_in+1:end-1]
    panel_consumption = panel_consumption[:,burn_in+1:end-1]
    shock_ρ = shock_ρ[:,burn_in+1:end-1]
    shock_e_1 = shock_e_1[:,burn_in+1:end-1]
    shock_e_2 = shock_e_2[:,burn_in+1:end-1]
    shock_e_3 = shock_e_3[:,burn_in+1:end-1]
    shock_ν = shock_ν[:,burn_in+1:end-1]

    # return results
    return panel_asset, panel_history, panel_default, panel_age, panel_consumption, shock_ρ, shock_e_1, shock_e_2, shock_e_3, shock_ν
end
