function itp_test_1(n)
    xs = collect(1:0.2:5)
    ny = 10
    A = [log(x)+y for x in xs, y in 1:ny]
    
    itp = linear_interpolation(xs, A[:,1], extrapolation_bc=Line())  # Lets call this the placeholder interpolation object
    
    # Replace the `coefs` field of the interpolation object with the new data
    for _ in 1:ny
        for iy = 1:ny
            @views itp.itp.coefs[:] = A[:,iy]
            # @assert itp(1.) == iy "$(itp(1.0)) == $(iy)"
            itp.(rand(n))
        end
    end
    nothing
end

function itp_test_2(n)
    xs = collect(1:0.2:5)
    ny = 10
    A = [log(x)+y for x in xs, y in 1:ny]
    
    itp = linear_interpolation(xs, A[:,1], extrapolation_bc=Line())  # Lets call this the placeholder interpolation object
    
    # Replace the `coefs` field of the interpolation object with the new data
    for _ in 1:ny
        for iy = 1:ny
            # @views itp.itp.coefs[:] = A[:,iy]
            # @assert itp(1.) == iy "$(itp(1.0)) == $(iy)"
            itp = linear_interpolation(xs, A[:,iy], extrapolation_bc=Line())
            itp.(rand(n))
        end
    end
    nothing
end

# function itp_test_3()
#     xs = collect(1:0.2:5)
#     ny = 10
#     A = [log(x)+y for x in xs, y in 1:ny]
    
#     itp = Akima(xs, A[:,1])  # Lets call this the placeholder interpolation object
    
#     # Replace the `coefs` field of the interpolation object with the new data
#     for _ in 1:ny
#         for iy = 1:ny
#             @views itp.ydata = A[:,iy]
#             # @assert itp(1.) == iy "$(itp(1.0)) == $(iy)"
#         end
#     end
#     nothing
# end

function itp_test_4(n)
    xs = collect(1:0.2:5)
    ny = 10
    A = [log(x)+y for x in xs, y in 1:ny]
    
    itp = Akima(xs, A[:,1])  # Lets call this the placeholder interpolation object
    
    # Replace the `coefs` field of the interpolation object with the new data
    for _ in 1:ny
        for iy = 1:ny
            itp = Akima(xs, A[:,iy])
            itp.(rand(n))
        end
    end
    nothing
end

using BenchmarkTools, Interpolations, FLOWMath

@btime itp_test_1(100)
@btime itp_test_2(100)
# @btime itp_test_3()
@btime itp_test_4(100)

@profview itp_test_4(100)
