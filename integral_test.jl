using QuadGK

f1(x) = exp(-x^2)
integral1, err1 = quadgk(x -> f1(x), 0, 1, rtol=1E-8)

f2(x) = (1/sqrt(2*π))*exp(-x^2/2.0)
integral2, err2 = quadgk(x -> f2(x), -Inf, Inf, rtol=1E-8)

f3(x) = x*(1/sqrt(2*π))*exp(-x^2/2.0)
integral3, err3 = quadgk(x -> f3(x), -Inf, Inf, order=100, rtol=1E-10)
