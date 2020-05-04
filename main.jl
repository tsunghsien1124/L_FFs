#------------------------------------------------------------------------------#
#                       Specify the Necessary Packages                         #
#------------------------------------------------------------------------------#

using ProgressMeter
using Parameters
using QuantEcon
using Plots
using Optim
using Interpolations


#------------------------------------------------------------------------------#
#              Define Parameters and Initialize Grids and Values               #
#------------------------------------------------------------------------------#


function RHS_bad(ap::Real, a_i::Integer, x_i::Integer, r::Real,
                 NR_parameters::NamedTuple,
                 EV_bad::Array{Float64,2}, EV_good::Array{Float64,2})

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # unpack shocks in the current period
    γ, p, t = x_grid[x_i,:]

    # compute consumption
    c = a_grid_pos[a_i]*(1+r) + p*t - ap

    # set up the expected value
    V_expec = 0.0
    V_update = 0.0

    if γ != 0.0
        for xp_i in 1:x_size
            # EV_bad_itp = interpolate((a_grid_pos,), EV_bad[:,xp_i], Gridded(Linear()))
            # EV_good_itp = interpolate((a_grid_pos,), EV_good[ind_a_zero:end,xp_i], Gridded(Linear()))
            EV_bad_itp = Spline1D(a_grid_pos, EV_bad[:,xp_i]; bc = "extrapolate")
            EV_good_itp = Spline1D(a_grid_pos, EV_good[ind_a_zero:end,xp_i]; bc = "extrapolate")
            V_expec += Px[x_i, xp_i]*( λ*EV_good_itp(ap) + (1-λ)*EV_bad_itp(ap) )
        end
        V_update = u(c,σ) + β*γ*V_expec
    else
        V_update = u(c,σ)
    end

    return V_update
end

function RHS_good_repay(ap::Real, a_i::Integer, x_i::Integer, r::Real,
                        NR_parameters::NamedTuple, NR_variables::NREconomy_variables,
                        EV_good::Array{Float64,2})

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # unpack shocks in the current period
    γ, p, t = x_grid[x_i,:]

    # compute comsumption
    # q_itp = interpolate((a_grid,), NR_variables.q[:,x_i], Gridded(Linear()))
    q_itp = Spline1D(a_grid, NR_variables.q[:,x_i]; bc = "extrapolate")
    c = a_grid[a_i]*( 1 + r*(a_grid[a_i]>0) ) + p*t - ap*q_itp(ap)

    # set up the expected value
    V_expec = 0.0
    V_update = 0.0

    if γ != 0.0
        for xp_i in 1:x_size
            # EV_good_itp = interpolate((a_grid,), EV_good[:,xp_i], Gridded(Linear()))
            EV_good_itp = Spline1D(a_grid, EV_good[:,xp_i]; bc = "extrapolate")
            V_expec += Px[x_i,xp_i]*EV_good_itp(ap)
        end
        V_update = u(c,σ) + β*γ*V_expec
    else
        V_update = u(c,σ)
    end

    return V_update
end

function RHS_good_default(x_i::Integer,
                          NR_parameters::NamedTuple, NR_variables::NREconomy_variables,
                          EV_bad::Array{Float64,2})

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # unpack shocks in the current period
    γ, p, t = x_grid[x_i,:]

    # compute comsumption
    c = p*t*(1-ξ)

    # set up the expected value
    V_expec = 0.0
    V_update = 0.0
    if γ != 0.0
        for xp_i in 1:x_size
            # EV_bad_itp = Spline1D(a_grid_pos, EV_bad[:,xp_i]; bc = "extrapolate")
            # V_expec += Px[x_i,xp_i]*EV_bad_itp(0.0)
            V_expec += Px[x_i,xp_i]*EV_bad[1,xp_i]
        end
        V_update = u(c,σ) + β*γ*V_expec
    else
        V_update = u(c,σ)
    end

    return V_update
end

function Maximize_bad!(a_i::Integer, x_i::Integer, r::Real,
                       NR_parameters::NamedTuple, NR_variables::NREconomy_variables,
                       EV_bad::Array{Float64,2}, EV_good::Array{Float64,2})

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters
    # unpack shocks in the current period
    γ, p, t = x_grid[x_i,:]

    # define the upper bound for the asset holding in the next period
    a_upper = min(a_grid_pos[a_i]*(1+r)+p*t, a_grid[end])

    # define objective functions
    obj_bad(ap) = -RHS_bad(ap,a_i,x_i,r,NR_parameters,EV_bad,EV_good)

    # maximize the objective function
    results_bad = optimize(obj_bad, 0.0, a_upper)

    # obtain results
    NR_variables.policy_a_bad[a_i,x_i] = Optim.minimizer(results_bad)
    NR_variables.V_bad[a_i,x_i] = -Optim.minimum(results_bad)
end

function Maximize_good_repay!(a_i::Integer, x_i::Integer, r::Real,
                              NR_parameters::NamedTuple, NR_variables::NREconomy_variables,
                              EV_good::Array{Float64,2})

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # unpack shocks in the current period
    γ, p, t = x_grid[x_i,:]

    # define the boundary for the asset holding in the next period
    a_upper = min(a_grid[a_i]*( 1 + r*(a_grid[a_i]>0) ) + p*t, a_grid[end])

    # compute the risky borrowing limit
    q_itp = Spline1D(a_grid, NR_variables.q[:,x_i]; bc = "extrapolate")
    obj_risky_borrowing_limit(ap) = ap*q_itp(ap)
    results_rbl = optimize(obj_risky_borrowing_limit, a_grid[1], a_grid[end])
    a_rbl = Optim.minimizer(results_rbl)

    if a_upper < a_rbl
        a_lower = a_grid[1]
    else
        a_lower = a_rbl
    end

    # define objective functions
    obj_good_repay(ap) = -RHS_good_repay(ap,a_i,x_i,r,NR_parameters,NR_variables,EV_good)

    # maximize the objective function
    results_good_repay = optimize(obj_good_repay, a_lower, a_upper)

    # obtain results
    NR_variables.policy_a_good_repay[a_i,x_i] = Optim.minimizer(results_good_repay)
    NR_variables.V_good_repay[a_i,x_i] = -Optim.minimum(results_good_repay)
end

function Maximize_good_default!(a_i::Integer, x_i::Integer,
                                NR_parameters::NamedTuple, NR_variables::NREconomy_variables,
                                EV_bad::Array{Float64,2})

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # obtain results
    NR_variables.V_good_default[a_i,x_i] = RHS_good_default(x_i, NR_parameters, NR_variables, EV_bad)
end

function Households!(r::Real,
                     NR_parameters::NamedTuple, NR_variables::NREconomy_variables,
                     EV_bad::Array{Float64,2}, EV_good::Array{Float64,2},
                     EV_good_default::Array{Float64,2}, EV_good_repay::Array{Float64,2})

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    for x_i in 1:x_size
        #---------------------------------------------------#
        # updating the value function of bad credit history #
        #---------------------------------------------------#
        for a_i in 1:a_size_pos
            # println("Maxizing the value function of bad credit history...")
            Maximize_bad!(a_i,x_i,r,NR_parameters,NR_variables,EV_bad,EV_good)
        end

        #----------------------------------------------------#
        # updating the value function of good credit history #
        #----------------------------------------------------#
        for a_i in 1:a_size
            # println("Maxizing the value function of good credit history and repaying...")
            Maximize_good_repay!(a_i,x_i,r,NR_parameters,NR_variables,EV_good)
            # println("Maxizing the value function of good credit history and defaulting...")
            Maximize_good_default!(a_i,x_i,NR_parameters,NR_variables,EV_bad)

            if NR_variables.V_good_default[a_i,x_i] > NR_variables.V_good_repay[a_i,x_i]
                NR_variables.V_good[a_i,x_i] = NR_variables.V_good_default[a_i,x_i]
                NR_variables.policy_a_good[a_i,x_i] = 0.0
            else
                NR_variables.V_good[a_i,x_i] = NR_variables.V_good_repay[a_i,x_i]
                NR_variables.policy_a_good[a_i,x_i] = NR_variables.policy_a_good_repay[a_i,x_i]
            end
        end
    end

    # update the policy matrices
    Policy_Matrix_bad!(NR_parameters, NR_variables)
    Policy_Matrix_good!(NR_parameters, NR_variables)
end

function Find_convex_combination(ap::Real, a_grid::Array{Float64,1})

    # find the indices
    ind_lower = maximum(findall(a_grid .<= ap))
    ind_upper = minimum(findall(a_grid .>= ap))

    if ind_lower == ind_upper
        coef_lower = 1.0
        coef_upper = 1.0
    else
        # find the weights of convex conbination
        coef_lower = (a_grid[ind_upper]-ap) / (a_grid[ind_upper]-a_grid[ind_lower])
        coef_upper = (ap-a_grid[ind_lower]) / (a_grid[ind_upper]-a_grid[ind_lower])
    end

    return (ind_lower, ind_upper, coef_lower, coef_upper)
end

function Policy_Matrix_good!(NR_parameters::NamedTuple, NR_variables::NREconomy_variables)

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # initialize the policy matrix
    NR_variables.policy_matrix_a_good_default = spzeros(a_size, a_size*x_size)
    NR_variables.policy_matrix_a_good_repay = spzeros(a_size, a_size*x_size)

    for x_i in 1:x_size, a_i in 1:a_size

        # choose to default
        if NR_variables.V_good_default[a_i,x_i] > NR_variables.V_good_repay[a_i,x_i]
            # find the associated index
            ind_ = (x_i-1)*a_size + ind_a_zero
            # fit in the coefficient
            NR_variables.policy_matrix_a_good_default[a_i,ind_] = 1.0

        # choose to repay
        else
            # find the optimal choice
            ap_good_repay = NR_variables.policy_a_good_repay[a_i,x_i]
            ind_lower_good_repay, ind_upper_good_repay, coef_lower_good_repay,
            coef_upper_good_repay = Find_convex_combination(ap_good_repay,a_grid)

            # find the associated indices
            ind_lower_ = (x_i-1)*a_size + ind_lower_good_repay
            ind_upper_ = (x_i-1)*a_size + ind_upper_good_repay

            # fit in the coefficients
            NR_variables.policy_matrix_a_good_repay[a_i,ind_lower_] = coef_lower_good_repay
            NR_variables.policy_matrix_a_good_repay[a_i,ind_upper_] = coef_upper_good_repay
        end
    end
end

function Policy_Matrix_bad!(NR_parameters::NamedTuple, NR_variables::NREconomy_variables)

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # initialize the policy matrix
    NR_variables.policy_matrix_a_bad = spzeros(a_size, a_size*x_size)

    for x_i in 1:x_size, a_i in 1:a_size_pos
        # find the optimal choice
        ap = NR_variables.policy_a_bad[a_i,x_i]
        ind_lower, ind_upper, coef_lower, coef_upper = Find_convex_combination(ap,a_grid)

        # find the associated indices
        ind_lower_ = (x_i-1)*a_size + ind_lower
        ind_upper_ = (x_i-1)*a_size + ind_upper

        # fit in the coefficients
        NR_variables.policy_matrix_a_bad[(a_size_neg+a_i),ind_lower_] = coef_lower
        NR_variables.policy_matrix_a_bad[(a_size_neg+a_i),ind_upper_] = coef_upper
    end
end

function Howard_Improvement!(r::Real ,NR_parameters::NamedTuple, NR_variables::NREconomy_variables; HI_iter::Integer=10)

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # initialize the updated objects
    EV_bad = similar(NR_variables.V_bad)
    EV_good = similar(NR_variables.V_good)
    EV_good_default = similar(NR_variables.V_good_default)
    EV_good_repay = similar(NR_variables.V_good_repay)

    # copy the previous values
    # copyto!(EV_bad, NR_variables.V_bad)
    # copyto!(EV_good, NR_variables.V_good)
    # copyto!(EV_good_default, NR_variables.V_good_default)
    # copyto!(EV_good_repay, NR_variables.V_good_repay)

    for i in 1:HI_iter
        for x_i in 1:x_size, a_i in 1:a_size
            if a_i > a_size_neg
                EV_bad[(a_i-a_size_neg),x_i] = RHS_bad(NR_variables.policy_a_bad[(a_i-a_size_neg),x_i], (a_i-a_size_neg), x_i, r, NR_parameters, NR_variables.V_bad, NR_variables.V_good)
            end
            EV_good_repay[a_i,x_i] = RHS_good_repay(NR_variables.policy_a_good_repay[a_i,x_i], a_i, x_i, r, NR_parameters, NR_variables, NR_variables.V_good)
            EV_good_default[a_i,x_i] = RHS_good_default(x_i, NR_parameters, NR_variables, NR_variables.V_bad)
        end

        copyto!(NR_variables.V_bad, EV_bad)
        copyto!(NR_variables.V_good_repay, EV_good_repay)
        copyto!(NR_variables.V_good_default, EV_good_default)

        for x_i in 1:x_size, a_i in 1:a_size
            if NR_variables.V_good_default[a_i,x_i] > NR_variables.V_good_repay[a_i,x_i]
                NR_variables.V_good[a_i,x_i] = NR_variables.V_good_default[a_i,x_i]
            else
                NR_variables.V_good[a_i,x_i] = NR_variables.V_good_repay[a_i,x_i]
            end
        end
    end
end

function Banks!(r::Real,
                NR_parameters::NamedTuple, NR_variables::NREconomy_variables)

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # initialize the pricing function
    NR_variables.q .= 0.0

    # update pricing function and default probability
    for x_i in 1:x_size
        for ap_i in 1:a_size_neg

            # initialize the expected revenue
            revenue_expect = 0.0

            for xp_i in 1:x_size
                # interpolate the value function of repaying
                # a_good_repay = cat(NR_variables.policy_a_good_repay[:,xp_i], NR_variables.V_good_repay[:,xp_i], dims=2)
                # ind_policy = sortperm(a_good_repay[:,1])
                # a_good_repay_sorted = a_good_repay[ind_policy,:]
                # V_good_repay_itp = Spline1D(a_good_repay_sorted[:,1], a_good_repay_sorted[:,2]; bc = "extrapolate")

                # unpack shocks in the next period
                γp, pp, tp = x_grid[xp_i,:]

                # choose to default
                if NR_variables.V_good_default[ap_i,xp_i] > NR_variables.V_good_repay[ap_i,xp_i]
                # if NR_variables.V_good_default[1,xp_i] > V_good_repay_itp(a_grid[ap_i])
                    revenue_expect += Px[x_i,xp_i]*ξ*pp*tp
                # choose to repay
                else
                    revenue_expect += Px[x_i,xp_i]*(-a_grid[ap_i])
                end
            end

            q_update = revenue_expect / ( (1+r)*(-a_grid[ap_i]) )
            NR_variables.q[ap_i,x_i] = q_update < 1 ? q_update : 1.0
        end
    end # end of loops over a and x'

    NR_variables.q[ind_a_zero:end,:] .= 1.0
end

function Stationary_Distribution!(NR_parameters::NamedTuple, NR_variables::NREconomy_variables,
                                  Eμ::Array{Float64,3})

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # initialize the cross-sectional distribution
    NR_variables.μ .= 0.0

    # select the part we need
    # μ_ = copyto!(NR_variables.μ
    # Eμ_ = copyto!(cat(reshape(Eμ[:,:,1], (:,1)), reshape(Eμ[ind_a_zero:end,:,2]), dims=2))

    # set up the aggregate policy matrix
    G_size_1 = a_size * x_size
    G_size_2 = a_size_pos * x_size
    G_size = G_size_1 + G_size_2
    NR_variables.transition_matrix = spzeros(G_size, G_size)

    # fit in numbers into the aggregate policy matrix (very sparse)
    for x_i in 1:x_size
        #---------------------#
        # good credit history #
        #---------------------#
        # define the associated indices
        row_begin = (x_i-1)*a_size + 1
        row_end = (x_i-1)*a_size + a_size

        col_begin_repay = 1
        col_end_repay = G_size_1

        col_begin_default = G_size_1 + 1
        col_end_default = G_size
        col_begin_default_temp = (x_i-1)*a_size + ind_a_zero
        col_end_default_temp = x_i*a_size

        # fit in numbers
        NR_variables.transition_matrix[row_begin:row_end,col_begin_repay:col_end_repay] = kron(transpose(Px[x_i,:]),NR_variables.policy_matrix_a_good_repay[:,row_begin:row_end])
        NR_variables.transition_matrix[row_begin:row_end,col_begin_default:col_end_default] = kron(transpose(Px[x_i,:]),NR_variables.policy_matrix_a_good_default[:,col_begin_default_temp:col_end_default_temp])

        #--------------------#
        # bad credit history #
        #--------------------#
        # define the associated indices
        row_begin_bad = G_size_1 + (x_i-1)*a_size_pos + 1
        row_end_bad = G_size_1 + (x_i-1)*a_size_pos + a_size_pos

        col_begin_erased = 1
        col_end_erased = G_size_1

        col_begin_remain = G_size_1 + 1
        col_end_remain = G_size
        col_begin_remain_temp = (x_i-1)*a_size + ind_a_zero
        col_end_remain_temp = x_i*a_size

        # fit in numbers
        NR_variables.transition_matrix[row_begin_bad:row_end_bad,col_begin_erased:col_end_erased] = λ*kron(transpose(Px[x_i,:]),NR_variables.policy_matrix_a_bad[ind_a_zero:end,row_begin:row_end])
        NR_variables.transition_matrix[row_begin_bad:row_end_bad,col_begin_remain:col_end_remain] = (1-λ)*kron(transpose(Px[x_i,:]),NR_variables.policy_matrix_a_bad[ind_a_zero:end,col_begin_remain_temp:col_end_remain_temp])
    end

    # compute the stationary distribution
    MC = MarkovChain(NR_variables.transition_matrix)
    SD = stationary_distributions(MC)

    # assign values
    NR_variables.μ[:,:,1] = SD[1][1:G_size_1]
    NR_variables.μ[ind_a_zero:end,:,2] = SD[1][(G_size_1+1):G_size]
end

function Excess_Demand!(NR_parameters::NamedTuple, NR_variables::NREconomy_variables)

    # unpack parameters
    @unpack λ, β, ξ, σ = NR_parameters
    # unpack stuff related to actions
    @unpack a_grid, ind_a_zero, a_size, a_size_pos, a_size_neg, a_grid_neg, a_grid_pos = NR_parameters
    # unpack idiosyncratic shocks
    @unpack Px, x_grid, x_size, x_ind = NR_parameters

    # compute aggregate deposits and liabilities
    NR_variables.D = 0.0
    NR_variables.L = 0.0

    for x_i in 1:x_size, a_i in 1:a_size

        # good credit history
        ap_good = NR_variables.policy_a_good[a_i,x_i]
        if ap_good > 0.0
            NR_variables.D += ap_good*NR_variables.μ[a_i,x_i,1]
        else
            # q_itp = interpolate((a_grid,), NR_variables.q[:,x_i], Gridded(Linear()))
            q_itp = Spline1D(a_grid, NR_variables.q[:,x_i]; bc = "extrapolate")
            NR_variables.L += -ap_good*q_itp(ap_good)*NR_variables.μ[a_i,x_i,1]
        end

        # bad credit history
        if a_i > a_size_neg
            ap_bad = NR_variables.policy_a_bad[(a_i-a_size_neg),x_i]
            NR_variables.D += ap_bad*NR_variables.μ[a_i,x_i,2]
        end
    end
end

function Solve_r!(r, NR_parameters;
                  tol_b = 1E-4, tol_h = 1E-4, tol_μ = 1E-8, iter_max = 30)

    # create the mutable objects
    NR_variables  = NREconomy_variables(r, NR_parameters)

    # update pring function with the initial guess of value functions
    # Banks!(r, NR_parameters, NR_variables)

    # initialize the iteration number and criterion for banks' problem
    iter_b = 0
    crit_b = Inf

    # initialize the updated pricing function
    Eq = similar(NR_variables.q)

    while crit_b > tol_b && iter_b < iter_max

        if iter_b == 0
            copyto!(Eq, NR_variables.q)
        else
            copyto!(Eq, 0.85*NR_variables.q .+ 0.15*Eq)
        end

        # initialize the iteration number and criterion for household's problem
        iter_h = 0
        crit_h = Inf
        prog_h = ProgressThresh(tol_h, "Solving household's maximization: ")

        # initialize the updated objects
        EV_bad = similar(NR_variables.V_bad)
        EV_good = similar(NR_variables.V_good)
        EV_good_default = similar(NR_variables.V_good_default)
        EV_good_repay = similar(NR_variables.V_good_repay)

        # println("1) Starting iterating household's maximization")
        while crit_h > tol_h && iter_h < iter_max

            # update the iteration number
            iter_h += 1

            # copy the previous values
            copyto!(EV_bad, NR_variables.V_bad)
            copyto!(EV_good, NR_variables.V_good)
            copyto!(EV_good_default, NR_variables.V_good_default)
            copyto!(EV_good_repay, NR_variables.V_good_repay)

            # update value functions
            # println("Maxizing hoesehold's problem (iteration $(iter_h) with the distance of $(crit_h))")
            Households!(r, NR_parameters, NR_variables, EV_bad, EV_good, EV_good_default, EV_good_repay)

            # update value function with the just obtained policy function
            Howard_Improvement!(r, NR_parameters, NR_variables)

            crit_h = max(norm(NR_variables.V_bad-EV_bad, Inf), norm(NR_variables.V_good-EV_good, Inf))
            ProgressMeter.update!(prog_h, crit_h)

        end # end of while loop over value functions

        # update the iteration number
        iter_b += 1

        # update pring function
        Banks!(r, NR_parameters, NR_variables)

        crit_b = norm(NR_variables.q-Eq, Inf)

        if iter_b % 1 == 0
            println("Pricing function updated (iteration $(iter_b) with the distance of $(crit_b))")
        end
    end # end of the while loop over pricing function

    # initialize the iteration number and criterion for Stationary distribution
    iter_μ = 0
    crit_μ = Inf
    prog_μ = ProgressThresh(tol_μ, "Solving stationary distribution: ")

    # initialize the stationary distribution
    Eμ = similar(NR_variables.μ)

    while crit_μ > tol_μ && iter_μ < iter_max

        # update the iteration number
        iter_μ += 1

        # copy the previous values
        copyto!(Eμ, NR_variables.μ)

        # update the cross-sectional distribution
        Stationary_Distribution!(NR_parameters, NR_variables, Eμ)

        crit_μ = norm(NR_variables.μ-Eμ, Inf)
        ProgressMeter.update!(prog_μ, crit_μ)

    end # end of while loop over stationary distribution

    # update the equlibrium interest rate
    Excess_Demand!(NR_parameters, NR_variables)

    # return excess demand
    ED = NR_variables.L - NR_variables.D
    println("-----------------------------------------------------------------")
    println("Current interest rate $(r) yields excess demand of $(ED)")
    println("-----------------------------------------------------------------")

    # return abs(ED)
    return (abs(ED), NR_variables)
end

# create the instance
NR_parameters = NREconomy_parameters()

# create the obtimized object
object_r(r) = Solve_r!(r, NR_parameters)

# set up the lower, upper, and initial value
# lower = 0.0 + 1E-3
# upper = (1 / NR_parameters.β) - 1 - 1E-4
# initial = (upper + lower) / 2

# solve for the equlibrium interest rate
# results = find_zero(object_r, [initial, upper])
# results = optimize(object_r, initial, upper)
ED_abs, NR_variables = object_r(0.0345815325628699)
