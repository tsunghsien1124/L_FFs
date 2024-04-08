using BenchmarkTools
using Test

n_i, n_j, n_k = 4, 5, 6;
n_t = n_i * n_j * n_k
n_threads = Threads.nthreads()

#===========#
# just loop #
#===========#

# loop with standard order
res_sl = zeros(n_i, n_j, n_k);
function simple_loop!(res_sl)
    for i = 1:n_i, j = 1:n_j, k = 1:n_k
        res_sl[i, j, k] += 1.0
    end
end
simple_loop!(res_sl);
@test all(res_sl .== 1.0)
@btime simple_loop!($res_sl)

# loop with standard order and no inbound check
res_sl_i = zeros(n_i, n_j, n_k);
function simple_loop_inbound!(res_sl_i)
    for i = 1:n_i, j = 1:n_j, k = 1:n_k
        @inbounds res_sl_i[i, j, k] += 1.0
    end
end
simple_loop_inbound!(res_sl_i);
@test all(res_sl_i .== 1.0)
@btime simple_loop_inbound!($res_sl_i)

# loop with reverse order
res_sl_r = zeros(n_i, n_j, n_k);
function simple_loop_reverse!(res_sl_r)
    for k = 1:n_k, j = 1:n_j, i = 1:n_i
        res_sl_r[i, j, k] += 1.0
    end
end
simple_loop_reverse!(res_sl_r);
@test all(res_sl_r .== 1.0)
@btime simple_loop_reverse!($res_sl_r)

# loop with reverse order and no inbound check
res_sl_r_i = zeros(n_i, n_j, n_k);
function simple_loop_reverse_inbound!(res_sl_r_i)
    for k = 1:n_k, j = 1:n_j, i = 1:n_i
        @inbounds res_sl_r_i[i, j, k] += 1.0
    end
end
simple_loop_reverse_inbound!(res_sl_r_i);
@test all(res_sl_r_i .== 1.0)
@btime simple_loop_reverse_inbound!($res_sl_r_i)

#==========================#
# loop with multi-theading #
#==========================#

# loop with standard order
res_sl_mt = zeros(n_i, n_j, n_k);
function simple_loop_mt!(res_sl_mt)
    Threads.@threads for i = 1:n_i 
        for j = 1:n_j, k = 1:n_k
            @inbounds res_sl_mt[i, j, k] += 1.0
        end
    end
end
simple_loop_mt!(res_sl_mt);
@test all(res_sl_mt .== 1.0)
@btime simple_loop_mt!($res_sl_mt)

# loop with reverse order
res_sl_r_mt = zeros(n_i, n_j, n_k);
function simple_loop_reverse_mt!(res_sl_r_mt)
    Threads.@threads for k = 1:n_k
        for j = 1:n_j, i = 1:n_i
            @inbounds res_sl_r_mt[i, j, k] += 1.0
        end
    end
end
simple_loop_reverse_mt!(res_sl_r_mt);
@test all(res_sl_r_mt .== 1.0)
@btime simple_loop_reverse_mt!($res_sl_r_mt)

#====================================#
# multiple loops with multi-theading #
#====================================#
# res_ml_r_mt = zeros(n_i, n_j, n_k);
res_ml_r_mt_vec = zeros(n_t);

# function get_index(t)
#     n_ii = floor(Int, t/n_j/n_k) + 1
#     n_jj = mod(floor(Int, t/n_k), n_j) + 1
#     n_kk = mod(t,n_k) + 1
#     return n_ii, n_jj, n_kk
# end
function multiple_loop_reverse_mt!(res_ml_r_mt_vec)
    Threads.@threads for t = 0:(n_t-1)
        # n_ii, n_jj, n_kk = get_index(t)
        n_ii = floor(Int, t/n_j/n_k) + 1
        n_jj = mod(floor(Int, t/n_k), n_j) + 1
        n_kk = mod(t,n_k) + 1
        # @inbounds res_ml_r_mt[n_ii, n_jj, n_kk] += 1.0
        @inbounds res_ml_r_mt_vec[t+1] = (n_ii-1)*n_j*n_k + (n_jj-1)*n_k + n_kk
    end
end
multiple_loop_reverse_mt!(res_ml_r_mt_vec);
res_ml_r_mt = reshape(res_ml_r_mt_vec, (n_k, n_j, n_i))
# @test all(res_ml_r_mt_vec .== 1.0)
@btime multiple_loop_reverse_mt!($res_ml_r_mt_vec)