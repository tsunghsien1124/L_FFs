
include("FLOWMath.jl")
using Optim
using Main.FLOWMath: Akima
using Calculus

w = collect(-5.0:10.0)
W = w.^2 .- 2*w .+ 1.0
W_itp = Akima(w,W)
f(x) = W_itp(x[1])
function g!(G, x)
    G[1] = derivative(f,x[1])
end

#=
f(x) = x[1]^2
function g!(G, x)
    G[1] = 2*x[1]
end
=#

lower = [2.0]
upper = [20.0]
initial_x = [8.0]
inner_optimizer = GradientDescent()

results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))

results = optimize(f, g!, lower, upper, initial_x, Fminbox(GradientDescent()))
