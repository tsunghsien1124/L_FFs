using Random123

@inline function pack_counter(agent::Integer, time::Integer, shock::Integer)::UInt64
    @assert agent  >= 0 && agent  < 2^20
    @assert time   >= 0 && time   < 2^20
    @assert shock  >= 0 && shock  < 2^20
    return (UInt64(agent)  << 44) | (UInt64(time) << 24) | UInt64(shock)
end

@inline function crn_uniform(agent::Integer, time::Integer, shock::Integer; 
    sd1::UInt64=0x0000000000000000, sd2::UInt64=0x0000000000000000)::Float64
    key = (sd1, sd2)
    rng = Philox4x(UInt64, key)
    set_counter!(rng, pack_counter(agent, time, shock))
    return rand(rng, Float64)
end

@inline function crn_uniform(agent::Integer, time::Integer, shock::Integer,
                             rng::Philox4x{UInt64, R}) where {R}
    set_counter!(rng, pack_counter(agent, time, shock))
    return rand(rng, Float64)
end

function make_thread_rngs(seed::Integer)
    key = (UInt64(seed), UInt64(0))
    return [Philox4x(UInt64, key) for _ in 1:Threads.nthreads()]
end

rngs = make_thread_rngs(1124)
u = crn_uniform(12, 12, 13, rngs[2])

u = crn_uniform(12, 12, 12; sd1=UInt64(1124))


function make_thread_rngs(seed::Integer)
    key = (UInt64(seed), UInt64(0))
    return [Philox4x(UInt64, key) for _ in 1:Threads.nthreads()]
end

using Polyester, Random123, BenchmarkTools

@inline base_counter(h_id::UInt64, t_id::UInt64)::UInt64 = (h_id << 44) | (t_id << 24)

@inline pack_counter(ht_id::UInt64, s::Integer)::UInt64 = ht_id | UInt64(s)

@inline function crn_uniform(ht_id::UInt64, s::Integer,
                             rng::Philox4x{UInt64, R}) where {R}
    set_counter!(rng, pack_counter(ht_id, s))
    return rand(rng, Float64)
end

function make_thread_rngs(seed::Integer, num_threads::Integer)
    key = (UInt64(seed), UInt64(0))
    return [Philox4x(UInt64, key) for _ in 1:num_threads]
end

function simulate_HH_panel(num_households::Int64=50000, num_periods::Int64=2000, seed::Int64=1124)

    @assert 0 < num_households <= 2^20  "The number of households exceeds 20-bit capacity"
    @assert 0 < num_periods    <= 2^20  "The number of periods exceeds 20-bit capacity"
    
    num_threads = Threads.nthreads()    
    rngs = make_thread_rngs(seed, num_threads)

    # Threads.@threads :static 
    @batch for h_i in 1:num_households
        thread_id = Threads.threadid()
        rng = rngs[thread_id]
        h_id = UInt64(h_i)
        for t_i in 1:num_periods
            t_id = UInt64(t_i)
            ht_id = base_counter(h_id, t_id)
            # println("$thread_id, $(crn_uniform.(ht_id, 1:2, rng))")
        end
    end
end

@btime simulate_HH_panel()