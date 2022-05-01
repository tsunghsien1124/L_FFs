#=============================#
# Solve transitional dynamics #
#=============================#

mutable struct Mutable_Aggregate_Prices_T
    """
    construct a type for mutable aggregate prices of periods T
    """
    λ::Array{Float64,1}
    ξ_λ::Array{Float64,1}
    Λ_λ::Array{Float64,1}
    leverage_ratio_λ::Array{Float64,1}
    KL_to_D_ratio_λ::Array{Float64,1}
    ι_λ::Array{Float64,1}
    r_k_λ::Array{Float64,1}
    K_p_λ::Array{Float64,1}
    w_λ::Array{Float64,1}
end

mutable struct Mutable_Aggregate_Variables_T
    """
    construct a type for mutable aggregate variables of periods T
    """
    K_p::Array{Float64,1}
    L_p::Array{Float64,1}
    D_p::Array{Float64,1}
    N::Array{Float64,1}
    leverage_ratio::Array{Float64,1}
    KL_to_D_ratio::Array{Float64,1}
    debt_to_earning_ratio::Array{Float64,1}
    share_of_filers::Array{Float64,1}
    share_of_involuntary_filers::Array{Float64,1}
    share_in_debts::Array{Float64,1}
    avg_loan_rate::Array{Float64,1}
    avg_loan_rate_pw::Array{Float64,1}
end

mutable struct Mutable_Variables_T
    """
    construct a type for mutable variables of periods T
    """
    aggregate_prices::Mutable_Aggregate_Prices_T
    aggregate_variables::Mutable_Aggregate_Variables_T
    R::Array{Float64,4}
    q::Array{Float64,4}
    rbl::Array{Float64,4}
    V::Array{Float64,6}
    V_d::Array{Float64,6}
    V_nd::Array{Float64,6}
    V_pos::Array{Float64,6}
    policy_a::Array{Float64,6}
    policy_d::Array{Float64,6}
    policy_pos_a::Array{Float64,6}
    μ::Array{Float64,7}
end

function aggregate_price_update(leverage_ratio_λ::Array{Float64,1}, variables_old::Mutable_Variables, variables_new::Mutable_Variables, parameters_new::NamedTuple)
    """
    update aggregate prices given a series of leverage ratio
    """

    @unpack a_size, a_size_pos, a_size_neg, a_size_μ, a_size_pos_μ, a_ind_zero_μ, e_1_size, e_2_size, e_3_size, ν_size, ρ, θ, ψ, r_f, E, δ, α = parameters_new

    T_size = length(leverage_ratio_λ)

    λ = zeros(T_size)
    λ[1] = variables_old.aggregate_prices.λ
    λ[end] = variables_new.aggregate_prices.λ

    ξ_λ = zeros(T_size)
    ξ_λ[1] = variables_old.aggregate_prices.ξ_λ
    ξ_λ[end] = variables_new.aggregate_prices.ξ_λ

    Λ_λ = zeros(T_size)
    Λ_λ[1] = variables_old.aggregate_prices.Λ_λ
    Λ_λ[end] = variables_new.aggregate_prices.Λ_λ

    KL_to_D_ratio_λ = zeros(T_size)
    KL_to_D_ratio_λ[1] = variables_old.aggregate_prices.KL_to_D_ratio_λ
    KL_to_D_ratio_λ[end] = variables_new.aggregate_prices.KL_to_D_ratio_λ

    ι_λ = zeros(T_size)
    ι_λ[1] = variables_old.aggregate_prices.ι_λ
    ι_λ[end] = variables_new.aggregate_prices.ι_λ

    r_k_λ = zeros(T_size)
    r_k_λ[1] = variables_old.aggregate_prices.r_k_λ
    r_k_λ[end] = variables_new.aggregate_prices.r_k_λ

    K_p_λ = zeros(T_size)
    K_p_λ[1] = variables_old.aggregate_prices.K_λ
    K_p_λ[end-1] = variables_old.aggregate_prices.K_λ
    K_p_λ[end] = variables_new.aggregate_prices.K_λ

    w_λ = zeros(T_size)
    w_λ[1] = variables_old.aggregate_prices.w_λ
    w_λ[2] = variables_old.aggregate_prices.w_λ
    w_λ[end] = variables_new.aggregate_prices.w_λ

    for T_i in (T_size-1):(-1):2
        λ[T_i] = max(1.0 - (1.0 - ψ + ψ * ξ_λ[T_i+1]) / (θ * leverage_ratio_λ[T_i]), 0.0)
        ξ_λ[T_i] = (1.0 - ψ + ψ * ξ_λ[T_i+1]) / (1.0 - λ[T_i])
        Λ_λ[T_i] = (1.0 - ψ + ψ * ξ_λ[T_i]) / (1.0 + r_f)
        KL_to_D_ratio_λ[T_i] = leverage_ratio_λ[T_i] / (leverage_ratio_λ[T_i] - 1.0)
        ι_λ[T_i] = θ * λ[T_i] / Λ_λ[T_i+1]
        r_k_λ[T_i] = r_f + ι_λ[T_i]
        K_p_λ[T_i] = E * ((r_k_λ[T_i] + δ) / α)^(1.0 / (α - 1.0))
        w_λ[T_i+1] = (1.0 - α) * (K_p_λ[T_i] / E)^α
    end

    # for T_i in (T_size-1):(-1):2
    #     ξ_λ[T_i] = θ * leverage_ratio_λ[T_i]
    #     λ[T_i] = 1.0 - (1.0 - ψ + ψ * ξ_λ[T_i+1]) / ξ_λ[T_i]
    #     Λ_λ[T_i] = (1.0 - ψ + ψ * ξ_λ[T_i]) / (1.0 + r_f)
    #     KL_to_D_ratio_λ[T_i] = leverage_ratio_λ[T_i] / (leverage_ratio_λ[T_i] - 1.0)
    #     ι_λ[T_i] = θ * λ[T_i] / Λ_λ[T_i+1]
    #     r_k_λ[T_i] = r_f + ι_λ[T_i]
    #     K_p_λ[T_i] = E * ((r_k_λ[T_i] + δ) / α)^(1.0 / (α - 1.0))
    #     w_λ[T_i+1] = (1.0 - α) * (K_p_λ[T_i] / E)^α
    # end

    return λ, ξ_λ, Λ_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_p_λ, w_λ
end

function variables_T_function(variables_old::Mutable_Variables, variables_new::Mutable_Variables, parameters_new::NamedTuple; T_size::Integer = 80, T_degree::Real = 1.0)
    """
    construct a mutable object containing endogenous variables of periods T
    """

    # unpack parameters from new steady state
    @unpack a_size, a_size_pos, a_size_neg, a_size_μ, a_size_pos_μ, a_ind_zero_μ, e_1_size, e_2_size, e_3_size, ν_size, ρ, r_f = parameters_new

    # compute adjusted time periods
    T_size = T_size + 2

    # define aggregate prices
    leverage_ratio_λ = variables_new.aggregate_variables.leverage_ratio .+ ((range(T_size - 1, stop = 0.0, length = T_size) / (T_size - 1)) .^ T_degree) .* (variables_old.aggregate_variables.leverage_ratio - variables_new.aggregate_variables.leverage_ratio)

    λ, ξ_λ, Λ_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_p_λ, w_λ = aggregate_price_update(leverage_ratio_λ, variables_old, variables_new, parameters_new)

    aggregate_prices = Mutable_Aggregate_Prices_T(λ, ξ_λ, Λ_λ, leverage_ratio_λ, KL_to_D_ratio_λ, ι_λ, r_k_λ, K_p_λ, w_λ)

    # define aggregate variables
    K_p = zeros(T_size)
    K_p[1] = variables_old.aggregate_variables.K
    K_p[end] = variables_new.aggregate_variables.K

    L_p = zeros(T_size)
    L_p[1] = variables_old.aggregate_variables.L
    L_p[end] = variables_new.aggregate_variables.L

    D_p = zeros(T_size)
    D_p[1] = variables_old.aggregate_variables.D
    D_p[end] = variables_new.aggregate_variables.D

    N = zeros(T_size)
    N[1] = variables_old.aggregate_variables.N
    N[2] = variables_old.aggregate_variables.N
    N[end] = variables_new.aggregate_variables.N

    leverage_ratio = zeros(T_size)
    leverage_ratio[1] = variables_old.aggregate_variables.leverage_ratio
    leverage_ratio[end] = variables_new.aggregate_variables.leverage_ratio

    KL_to_D_ratio = zeros(T_size)
    KL_to_D_ratio[1] = variables_old.aggregate_variables.KL_to_D_ratio
    KL_to_D_ratio[end] = variables_new.aggregate_variables.KL_to_D_ratio

    debt_to_earning_ratio = zeros(T_size)
    debt_to_earning_ratio[1] = variables_old.aggregate_variables.debt_to_earning_ratio
    debt_to_earning_ratio[end] = variables_new.aggregate_variables.debt_to_earning_ratio

    share_of_filers = zeros(T_size)
    share_of_filers[1] = variables_old.aggregate_variables.share_of_filers
    share_of_filers[end] = variables_new.aggregate_variables.share_of_filers

    share_of_involuntary_filers = zeros(T_size)
    share_of_involuntary_filers[1] = variables_old.aggregate_variables.share_of_involuntary_filers
    share_of_involuntary_filers[end] = variables_new.aggregate_variables.share_of_involuntary_filers

    share_in_debts = zeros(T_size)
    share_in_debts[1] = variables_old.aggregate_variables.share_in_debts
    share_in_debts[end] = variables_new.aggregate_variables.share_in_debts

    avg_loan_rate = zeros(T_size)
    avg_loan_rate[1] = variables_old.aggregate_variables.avg_loan_rate
    avg_loan_rate[end] = variables_new.aggregate_variables.avg_loan_rate

    avg_loan_rate_pw = zeros(T_size)
    avg_loan_rate_pw[1] = variables_old.aggregate_variables.avg_loan_rate_pw
    avg_loan_rate_pw[end] = variables_new.aggregate_variables.avg_loan_rate_pw

    aggregate_variables = Mutable_Aggregate_Variables_T(K_p, L_p, D_p, N, leverage_ratio, KL_to_D_ratio, debt_to_earning_ratio, share_of_filers, share_of_involuntary_filers, share_in_debts, avg_loan_rate, avg_loan_rate_pw)

    # define repayment probability, pricing function, and risky borrowing limit
    R = zeros(a_size_neg, e_1_size, e_2_size, T_size)
    R[:,:,:,1] = variables_old.R
    R[:,:,:,end] = variables_new.R

    q = ones(a_size, e_1_size, e_2_size, T_size) .* ρ ./ (1.0 + r_f)
    q[:,:,:,1] = variables_old.q
    q[:,:,:,end] = variables_new.q

    rbl = zeros(e_1_size, e_2_size, 2, T_size)
    rbl[:,:,:,1] = variables_old.rbl
    rbl[:,:,:,end] = variables_new.rbl

    # define value and policy functions
    V = zeros(a_size, e_1_size, e_2_size, e_3_size, ν_size, T_size)
    V[:,:,:,:,:,1] = variables_old.V
    V[:,:,:,:,:,end] = variables_new.V

    V_d = zeros(a_size, e_1_size, e_2_size, e_3_size, ν_size, T_size)
    V_d[:,:,:,:,:,1] = variables_old.V_d
    V_d[:,:,:,:,:,end] = variables_new.V_d

    V_nd = zeros(a_size, e_1_size, e_2_size, e_3_size, ν_size, T_size)
    V_nd[:,:,:,:,:,1] = variables_old.V_nd
    V_nd[:,:,:,:,:,end] = variables_new.V_nd

    V_pos = zeros(a_size_pos, e_1_size, e_2_size, e_3_size, ν_size, T_size)
    V_pos[:,:,:,:,:,1] = variables_old.V_pos
    V_pos[:,:,:,:,:,end] = variables_new.V_pos

    policy_a = zeros(a_size, e_1_size, e_2_size, e_3_size, ν_size, T_size)
    policy_a[:,:,:,:,:,1] = variables_old.policy_a
    policy_a[:,:,:,:,:,end] = variables_new.policy_a

    policy_d = zeros(a_size, e_1_size, e_2_size, e_3_size, ν_size, T_size)
    policy_d[:,:,:,:,:,1] = variables_old.policy_d
    policy_d[:,:,:,:,:,end] = variables_new.policy_d

    policy_pos_a = zeros(a_size_pos, e_1_size, e_2_size, e_3_size, ν_size, T_size)
    policy_pos_a[:,:,:,:,:,1] = variables_old.policy_pos_a
    policy_pos_a[:,:,:,:,:,end] = variables_new.policy_pos_a

    # define cross-sectional distribution
    μ = zeros(a_size_μ, e_1_size, e_2_size, e_3_size, ν_size, 2, T_size)
    μ_size = (a_size_μ + a_size_pos_μ) * e_1_size * e_2_size * e_3_size * ν_size
    μ[:,:,:,:,:,1,2:(end-1)] .= 1.0 ./ μ_size
    μ[a_ind_zero_μ:end,:,:,:,:,2,2:(end-1)] .= 1.0 ./ μ_size
    μ[:,:,:,:,:,:,1] = variables_old.μ
    μ[:,:,:,:,:,:,end] = variables_new.μ

    # return outputs
    variables_T = Mutable_Variables_T(aggregate_prices, aggregate_variables, R, q, rbl, V, V_d, V_nd, V_pos, policy_a, policy_d, policy_pos_a, μ)
    return variables_T
end

function transitional_dynamic_λ_function!(variables_T::Mutable_Variables_T, variables_old::Mutable_Variables, variables_new::Mutable_Variables, parameters_new::NamedTuple; tol::Real = 1E-3, iter_max::Real = 500, slow_updating::Real = 1.0, figure_track::Bool = false)
    """
    solve transitional dynamics of periods T from initial to new steady states
    """

    # unpack parameters
    @unpack θ, ψ, r_f, E, δ, α = parameters_new
    T_size = size(variables_T.V)[end]

    # initialize the iteration number and criterion
    search_iter = 0
    crit = Inf

    # construct container
    leverage_ratio_λ_p = similar(variables_T.aggregate_prices.leverage_ratio_λ)

    while crit > tol && search_iter < iter_max

        # copy previous value
        copyto!(leverage_ratio_λ_p, variables_T.aggregate_prices.leverage_ratio_λ)

        # solve individual-level problems backward
        for T_i = (T_size-1):(-1):2

            # report progress
            # println("Solving individual-level problems backward... period $(T_i-1) / $(T_size-2)")

            # pricing function and borrowing risky limit
            variables_T.R[:,:,:,T_i], variables_T.q[:,:,:,T_i], variables_T.rbl[:,:,:,T_i] = pricing_and_rbl_function(variables_T.policy_d[:,:,:,:,:,T_i+1], variables_T.aggregate_prices.w_λ[T_i+1], variables_T.aggregate_prices.ι_λ[T_i], parameters_new)

            # value and policy functions
            variables_T.V[:,:,:,:,:,T_i], variables_T.V_d[:,:,:,:,:,T_i], variables_T.V_nd[:,:,:,:,:,T_i], variables_T.V_pos[:,:,:,:,:,T_i], variables_T.policy_a[:,:,:,:,:,T_i], variables_T.policy_d[:,:,:,:,:,T_i], variables_T.policy_pos_a[:,:,:,:,:,T_i] = value_and_policy_function(variables_T.V[:,:,:,:,:,T_i+1], variables_T.V_d[:,:,:,:,:,T_i+1], variables_T.V_nd[:,:,:,:,:,T_i+1], variables_T.V_pos[:,:,:,:,:,T_i+1], variables_T.q[:,:,:,T_i], variables_T.rbl[:,:,:,T_i], variables_T.aggregate_prices.w_λ[T_i], parameters_new)
        end

        # solve distribution forward and update aggregate variables and prices

        for T_i = 2:(T_size-1)

            # report progress
            # println("Solving distribution and aggregate variables/prices forward... period $(T_i-1) / $(T_size-2)")

            # update stationary distribution
            variables_T.μ[:,:,:,:,:,:,T_i] = stationary_distribution_function(variables_T.μ[:,:,:,:,:,:,T_i-1], variables_T.policy_a[:,:,:,:,:,T_i], variables_T.policy_d[:,:,:,:,:,T_i], variables_T.policy_pos_a[:,:,:,:,:,T_i], parameters_new)

            # compute aggregate variables
            aggregate_variables = solve_aggregate_variable_function(variables_T.policy_a[:,:,:,:,:,T_i], variables_T.policy_d[:,:,:,:,:,T_i], variables_T.policy_pos_a[:,:,:,:,:,T_i], variables_T.q[:,:,:,T_i], variables_T.rbl[:,:,:,T_i], variables_T.μ[:,:,:,:,:,:,T_i], variables_T.aggregate_prices.K_p_λ[T_i], variables_T.aggregate_prices.w_λ[T_i], variables_T.aggregate_prices.ι_λ[T_i], parameters_new)

            variables_T.aggregate_variables.K_p[T_i] = aggregate_variables.K
            variables_T.aggregate_variables.L_p[T_i] = aggregate_variables.L
            variables_T.aggregate_variables.D_p[T_i] = aggregate_variables.D
            variables_T.aggregate_variables.N[T_i+1] = variables_new.aggregate_variables.ω * ψ * aggregate_variables.profit

            variables_T.aggregate_variables.leverage_ratio[T_i] = (variables_T.aggregate_variables.K_p[T_i] + variables_T.aggregate_variables.L_p[T_i]) / variables_T.aggregate_variables.N[T_i]

            variables_T.aggregate_variables.KL_to_D_ratio[T_i] = aggregate_variables.KL_to_D_ratio

            variables_T.aggregate_variables.debt_to_earning_ratio[T_i] = aggregate_variables.debt_to_earning_ratio

            variables_T.aggregate_variables.share_of_filers[T_i] = aggregate_variables.share_of_filers
            variables_T.aggregate_variables.share_of_involuntary_filers[T_i] = aggregate_variables.share_of_involuntary_filers
            variables_T.aggregate_variables.share_in_debts[T_i] = aggregate_variables.share_in_debts

            variables_T.aggregate_variables.avg_loan_rate[T_i] = aggregate_variables.avg_loan_rate
            variables_T.aggregate_variables.avg_loan_rate_pw[T_i] = aggregate_variables.avg_loan_rate_pw
        end

        # check convergence
        crit = norm(variables_T.aggregate_variables.leverage_ratio .- leverage_ratio_λ_p, Inf)

        # update the iteration number
        search_iter += 1

        # manually report convergence progress
        println("Solving transitional dynamics: search_iter = $search_iter and crit = $crit > tol = $tol")

        # update leverage ratio
        variables_T.aggregate_prices.leverage_ratio_λ = (1.0 - slow_updating) * leverage_ratio_λ_p + slow_updating * variables_T.aggregate_variables.leverage_ratio

        # update aggregate prices
        variables_T.aggregate_prices.λ, variables_T.aggregate_prices.ξ_λ, variables_T.aggregate_prices.Λ_λ, variables_T.aggregate_prices.KL_to_D_ratio_λ, variables_T.aggregate_prices.ι_λ, variables_T.aggregate_prices.r_k_λ, variables_T.aggregate_prices.K_p_λ, variables_T.aggregate_prices.w_λ = aggregate_price_update(variables_T.aggregate_prices.leverage_ratio_λ, variables_old, variables_new, parameters_new)

        # tracking figures
        if figure_track == true
            println()
            plt_LR = lineplot(
                collect(2:(T_size-1)),
                variables_T.aggregate_variables.leverage_ratio[2:(T_size-1)],
                name = "updated",
                title = "leverage ratio",
                xlim = [0.0, T_size],
                width = 50,
                height = 10,
            )
            lineplot!(plt_LR, collect(2:(T_size-1)), leverage_ratio_λ_p[2:(T_size-1)], name = "initial")
            lineplot!(plt_LR, collect(2:(T_size-1)), variables_T.aggregate_prices.leverage_ratio_λ[2:(T_size-1)], name = "slow updating")
            println(plt_LR)
        end
    end
end
