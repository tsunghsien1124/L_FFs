function tauchenmethod(μ, σ, ρ, size, range; Verbose = 0)
  #return states, transmatrix

  # z' = ρ*z + e, e ~ N(μ,σ^2), by Tauchens method
  # q: max number of std devs from mean
  # znum: number of states in discretization of z (must be an odd number)

  if Verbose == 1
    println("Approximating an AR(1) process using Tauchen Method with $size points and q=$q.")
  end

  sigma = sqrt(sigmasq); #stddev of e
  zstar = mew/(1-rho); #expected value of z
  sigmaz = sigma/sqrt(1-rho^2); #stddev of z

  z = zstar*ones(znum,1) + linspace(-q*sigmaz,q*sigmaz,znum);
  omega = z[2]-z[1]; #Note that all the points are equidistant by construction.

  zi=z*ones(1,znum);
  zj=ones(znum,1)*z';

  P_part1=cdf(Normal(),((zj+omega/2-rho*zi)-mew)/sigma);
  P_part2=cdf(Normal(),((zj-omega/2-rho*zi)-mew)/sigma);

  P=P_part1-P_part2;
  P[:,1]=P_part1[:,1];
  P[:,znum]=1-P_part2[:,znum];


  #states=z;
  #transmatrix=P; #(z,zprime)
  return z, P

end
