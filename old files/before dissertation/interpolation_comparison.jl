using Distributions
using Plots
μ = 0.0
σ = 0.2
x = collect(-2.4:0.4:0.0)
p = 1.0 .- cdf.(LogNormal(μ,σ),-x)
p_alternative = 1.0 .- cdf.(Normal(μ,σ),log.(-x))

# Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
include("SimplePCHIP.jl")
using Main.SimplePCHIP
p_itp_PCHIP = interpolate(x, p)
p_itp_PCHIP_ = p_itp_PCHIP.(x)
plot(x, p_itp_PCHIP_, legend=:none, title="PCHIP")
plot!(x, p, legend=:none, seriestype=:scatter)

# Cubic Hermite Spline (CHS)
using CubicHermiteSpline
g = diff(p)./diff(x)
g = cat(g[1],g,dims=1)
# g = cat(g,g[end],dims=1)
p_itp_CHS = CubicHermiteSplineInterpolation(x, p, g)
p_itp_CHS_ = p_itp_CHS.(x)
plot(x, p_itp_CHS_, legend=:none, title="CHS")
plot!(x, p, legend=:none, seriestype=:scatter)
