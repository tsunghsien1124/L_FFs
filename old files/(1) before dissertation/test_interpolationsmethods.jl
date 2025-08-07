
include("SimplePCHIP.jl")
include("FLOWMath.jl")

using Plots
using Dierckx: Spline1D
using Interpolations: LinearInterpolation
using Main.SimplePCHIP
using Calculus
using Optim
using Main.FLOWMath: interp2d, akima, Akima


x = [-5.; -4.; -3.; -2.; -1.; 0.; 1.; 2.; 3.; 4.; 5.]
y = [9.; 9.; 9.; 4.; 1.; 0.; 1.; 4.; 9.; 9.; 9.]
# y = [0.; 0.; 0.; 0.; 0.; 0.; 1.; 4.; 9.; 16.; 25.]

itp_pchip = interpolate(x, y)
itp_cubicspline = Spline1D(x, y, k = 3)
itp_linear = LinearInterpolation(x, y)
itp_akima = Akima(x,y)

xs = -5:0.01:5
ys_pchip = itp_pchip.(xs)
ys_cubicspline = itp_cubicspline.(xs)
ys_linear = itp_linear.(xs)
ys_akima = itp_akima.(xs)
ys = [ys_pchip ys_cubicspline ys_linear ys_akima]

plot(legend =:bottomright)
plot!(x, y, seriestype=:scatter, label = "Points")
plot!(xs, ys, label = ["PCHIP" "Cubic" "Linear" "Akima"])

object_pchip(x_min) = itp_pchip(x_min)
object_cubicspline(x_min) = itp_cubicspline(x_min)
object_linear(x_min) = itp_linear(x_min)
object_akima(x_min) = itp_akima(x_min)

res_pchip = optimize(object_pchip, -4.0, 1.0)
res_cubicspline = optimize(object_cubicspline, -4.0, 1.0)
res_linear = optimize(object_linear, -4.0, 1.0)
res_akima = optimize(object_akima, -4.0, 1.0)

res_min = [Optim.minimizer(res_pchip), Optim.minimizer(res_cubicspline), Optim.minimizer(res_linear), Optim.minimizer(res_akima)]


x_dev = -0.0
dev_pchip = derivative(itp_pchip, x_dev)
dev_cubicspline = derivative(itp_cubicspline, x_dev)
dev_linear = derivative(itp_linear, x_dev)
dev_akima = derivative(itp_akima, x_dev)
res_dev = [dev_pchip, dev_cubicspline, dev_linear, dev_akima]
