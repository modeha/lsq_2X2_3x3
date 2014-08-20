function [x,IterN,realTol,objtrue,time]= pdco_l1ls(m,n)
Name = m;
nc = m;
 % spiky signal generation
 T = 10; % number of spikes
 x0 = zeros(n,1);
 q = randperm(n);
 x0(q(1:T)) = sign(randn(T,1));

 % noisy observations
 sigma = 0.01; % noise standard deviation
 %x0 = sprandvec(m,sparse);
 Q = dct(rand(m,n));
 y = Q*x0 + sigma*randn(m,1);
options = pdcoSet;

 
 M = m+2*n;
 N = 4*n;
 
 
 delta = 0.001;
 c = vertcat(zeros(3*n,1),delta*ones(n,1));
 b   = vertcat(y,zeros(2*n,1));
 
 I = eye(n);
 A = zeros(M,N);
 
 A(1:m,1:n)             = Q;
 A(m+1:m+n,1:n)         = -I;
 A(m+n+1:end,1:n)       = I;
 
 A(m+1:m+n,n+1:2*n)     = I;
 
 A(m+n+1:end,2*n+1:3*n) = I;
 
 A(m+1:m+n,3*n+1:end)   = I; 
 A(m+n+1:end,3*n+1:end) = -I;
 
  

  x0    = 0.5*ones(N,1);     % Initial x
  y0    = zeros(M,1);        % Initial y
  z0    = ones(N,1);         % Initial z
  xsize = 1;                 % Estimate of norm(x,inf) at solution
  zsize = 1;                 % Estimate of norm(z,inf) at solution

  options.mu0       = 1e-0;  % An absolute value
  options.LSMRatol1 = 1e-8;  % For LPs, LSQR must solve quite accurately
  options.wait      = 0;     % Allow options to be reviewed before solve
  options.Method    = 3;  % Will change to 1 or 3
  
 
  em      = ones(M,1);
  en      = ones(N,1);
  zn      = zeros(n,1);
  on      = ones(n,1);
  bigbnd  = 1e+30;
  inf = bigbnd*on;
  
  
  bl      = [-inf;zn;zn;zn]; 
  bu      = [inf;inf;inf;inf];

  gamma   = 1e-3;       % Primal regularization.
  delta   = 1e-3;       % 1e-3 or 1e-4 for LP;  1 for Least squares.
  d1      = gamma;      % Scalar for this test.
  d2      = em;         % D2 = I
  nc = N/10;
 % d2(M-nc+1:M) = delta; % Satisfy last nc rows more accurately.
%[x,IterN,realTol,objtrue,time] = ...
options
    pdco(c,A,b,bl,bu,d1,d2,options,x0,y0,z0,xsize,zsize);
end