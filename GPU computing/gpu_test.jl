#==============#
# Introduction #
#==============#
#=================#
# Install Package #
#=================#
# install the package
using Pkg
Pkg.add("CUDA")

# smoke test (this will download the CUDA toolkit)
using CUDA
CUDA.versioninfo()

#=================#
# CPU Computation #
#=================#
N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

y .+= x             # increment each element of y with the corresponding element of x

using Test
@test all(y .== 3.0f0)

function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
@test all(y .== 3.0f0)

function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parallel_add!(y, x)
@test all(y .== 3.0f0)

using BenchmarkTools
@btime sequential_add!($y, $x)
@btime parallel_add!($y, $x)

#=================#
# GPU Computation #
#=================#
using CUDA

x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

y_d .+= x_d.^2.0f0
@test all(Array(y_d) .== 3.0f0)

function add_broadcast!(y, x)
    CUDA.@sync y .+= x.^2.0f0
    return
end

add_broadcast!(y_d, x_d)
@btime add_broadcast!($y_d, $x_d)

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

@btime bench_gpu1!($y_d, $x_d)

bench_gpu1!(y_d, x_d)  # run it once to force compilation
CUDA.@profile bench_gpu1!(y_d, x_d)

CUDA.@profile trace=true bench_gpu1!(y_d, x_d)

function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda threads=256 gpu_add2!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu2!(y, x)
    CUDA.@sync begin
        @cuda threads=256 gpu_add2!(y, x)
    end
end

@btime bench_gpu2!($y_d, $x_d)

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

numblocks = ceil(Int, N/256)

fill!(y_d, 2)
@cuda threads=256 blocks=numblocks gpu_add3!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu3!(y, x)
    numblocks = ceil(Int, length(y)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks gpu_add3!(y, x)
    end
end

@btime bench_gpu3!($y_d, $x_d)

CUDA.@profile trace=true bench_gpu3!(y_d, x_d)

kernel = @cuda launch=false gpu_add3!(y_d, x_d)
config = launch_configuration(kernel.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)

fill!(y_d, 2)
kernel(y_d, x_d; threads, blocks)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu4!(y, x)
    kernel = @cuda launch=false gpu_add3!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync begin
        kernel(y, x; threads, blocks)
    end
end

@btime bench_gpu4!($y_d, $x_d)

function gpu_add2_print!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    @cuprintln("thread $index, block $stride")
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

@cuda threads=32 gpu_add2_print!(y_d, x_d)
synchronize()

#======================#
# Using custom structs #
#======================#
using CUDA

struct Interpolate{A}
    xs::A
    ys::A
end

function (itp::Interpolate)(x)
    i = searchsortedfirst(itp.xs, x)
    i = clamp(i, firstindex(itp.ys), lastindex(itp.ys))
    @inbounds itp.ys[i]
end

xs_cpu = [1.0, 2.0, 3.0]
ys_cpu = [10.0,20.0,30.0]
itp_cpu = Interpolate(xs_cpu, ys_cpu)
pts_cpu = [1.1,2.3]
result_cpu = itp_cpu.(pts_cpu)

# move to gpu
itp = Interpolate(CuArray(xs_cpu), CuArray(ys_cpu))
pts = CuArray(pts_cpu);
# itp.(pts) # error!
isbitstype(typeof(itp))

import Adapt
function Adapt.adapt_structure(to, itp::Interpolate)
    xs = Adapt.adapt_structure(to, itp.xs)
    ys = Adapt.adapt_structure(to, itp.ys)
    Interpolate(xs, ys)
end
Adapt.@adapt_structure Interpolate

result = itp.(pts)
@assert CuArray(result_cpu) == result
