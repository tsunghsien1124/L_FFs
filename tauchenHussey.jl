####################
#
# 	This function computes a discrete
# 	approximation to the AR(1) Auto Regressive Stochastic Process
#
#	y_t = (1-rho)*mu + rho*y_{t-1} + epsilon_t
#
#	where epsilon is normally distributed mean 0 and standard deviation sigma.
#   This is done using Tauchen and Hussey (1991)'s algorithm.
#
#   Note: you need the Distributions package.
#   If you don't already have it, install it by
#   Pkg.add("Distributions")
#   Load it by:   using Distributions
#
#   Inputs:
# 	n is the number of discrete points you want the approximation to use
#   rho is the value of the persistance parameter
#   sigma is the standard deviation of the innovation epsilon_t
#   mmu is mean of AR(1) process. Defaults to zero mean
#
#   Outputs:
#   actualPoints: length n vector of the points of the Normal Distribution
#   probabilities:  nxn matrix of transitions: entry   p_ij  tells you the probility of transitioning
#					from state i to state j.
#
#	Reference: http://www.fperri.net/TEACHING/macrotheory08/numerical.pdf
#
#   This code is a translation of the code by Martin Floden of Stockholm School of Economics
#   It can be found in: http://www2.hhs.se/personal/floden/CODE/tauchenhussey.m (Note that it has a mistake in the gausshermite loop (floor function)
####################



function tauchenHussey(n::Int64,rho::Float64,sigma::Float64;mmu::Float64=0.0) #

	if n < 2
		error("Only intended for at least n>2 nodes")
	elseif rho < -1 || rho > 1
		error("The process needs to be covariance stationary")
	elseif sigma < 0
		error("Provided a negative standard deviation for the epsilon")
	end

	transition = Array(Float64,n,n)

	w = 0.5 + rho/4

	baseSigma = w*sigma + (1-w)*sigma/(1-rho^2)


	actualPoints,weights = gaussnorm(n,mmu,baseSigma^2)
	# Distributions for computing transition
	dist = Distributions.Normal(0.0,sigma)
	distUncond = Distributions.Normal(0.0,baseSigma)

	for i in 1:n
		for j in 1:n              #dist centered in       # the deviation
			transProb  = pdf(dist,actualPoints[j] - rho*actualPoints[i] - (1-rho)*mmu)
			uncondProb = pdf(distUncond,actualPoints[j] - mmu) # no 1-rho as just unconditional
			transition[i,j] = weights[j]*(transProb/uncondProb)
		end
	end

	probabilities = Array(Float64,n,n)

	for i in 1:n
		for j in 1:n
			probabilities[i,j] = transition[i,j]/sum(transition[i,:])
		end
	end

	return actualPoints, probabilities
end

function gaussnorm(n,mmu,s2)
	# Find Gaussing nods and weights for the normal distribution
	x0,w0 = qnwnorm1(n)

	x = x0*sqrt(2.0*s2) + mmu
	w = w0/sqrt(pi)

	x,w
end

function qnwnorm1(n)

	maxit = 100;
	pim4 = 1/pi^.25
	m = floor((n+1)/2)
	x = zeros(n,1)
	w = zeros(n,1)
	z = 1.0; pp = 1.0;
	for i = 1:m
		if i==1 # Reasonable starting values
			z = sqrt(2*n+1)-1.85575*((2*n+1)^(-1/6));
		elseif i==2
			z = z-1.14*(n^0.426)/z;
		elseif i==3
			z = 1.86*z+0.86*x[1];
		elseif	i==4
			z = 1.91*z+0.91*x[2];
		else
			 z = 2*z+x[i-2];
		end
		its = 0;  # root finding iterations
		while its < maxit
			its = its+1;
	      	p1 = pim4;
	      	p2 = 0;
	      	for j=1:n
	         	p3 = p2;
	         	p2 = p1;
	         	p1 = z*sqrt(2/j)*p2-sqrt((j-1)/j)*p3;
	      	end;
	      	pp = sqrt(2*n)*p2;
	      	z1 = z;
	      	z  = z1-p1./pp;
	      	if abs(z-z1)<1e-14;
	      		break;
	      	end
	    end

	    if its>=maxit
	    	error("failed to converge in qnwnorm1")
	    end
	    x[n+1-i] = z;
	    x[i] = -z;
	    w[i] = 2.0/(pp*pp);
    	w[n+1-i] = w[i];
	end

	squeeze(x,2),squeeze(w,2)

end
