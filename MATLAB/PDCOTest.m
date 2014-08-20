function PDCOTest(m,n,k)

%Example of use:
% PDCOTest(128,1024,1)
% PDCOTest(128,1024,2)

 [A,y] = PDCO_partialDCT( m,n );
 %[A,y]= randPDCO(m,n);
 
 
  options = pdcoSet;
  options.StepTol = 0.5;
  options.FeaTol       =  1e-6;
  options.OptTol       =  1e-6;
  %options.MaxIter = 100;
  options.mu0       = 1e-0;  % An absolute value
  options.LSMRatol1 = 1e-6;  % For LPs, LSQR must solve quite accurately
  options.wait      = 0;     % Allow options to be reviewed before solve
  options.Print     = 1;
  options.Method    = 3;  % Will change to 1 or 3
                          %1=Cholesky  2=QR  3=LSMR  4=MINRES  21=SQD
  options.MaxIter      =    30;


  x0    = 0.5*ones(n,1);     % Initial x
  y0    = zeros(m,1);        % Initial y
  z0    = ones(n,1);         % Initial z
  xsize = 1;                 % Estimate of norm(x,inf) at solution
  zsize = 1;                 % Estimate of norm(z,inf) at solution

  c       = zeros(n,1);
  b       = y;
  em      = ones(m,1);
  zn      = zeros(n,1);
  bigbnd  = 1e+30;
  bl      = zn;         % x > 0
  bu      = bigbnd;%10*en;      % x < 10 where en      = ones(n,1);
%   bl(1)   = - bigbnd;   % Test "free variable" (no bounds)
%   bu(1)   =   bigbnd;
  nc = m;
  gamma   = 1;       % Primal regularization.
  delta   = 1e-3;       % 1e-3 or 1e-4 for LP;  1 for Least squares.
  d1      = gamma;      % Scalar for this test.
  d2      = em;         % D2 = I
  d2(m-nc+1:m) = delta; % Satisfy last nc rows more accurately.
  
  lsmr_solver = k;% lsmr_solver = 1 use PDCO lsmr, lsmr_solver = 2 use lsmr_spot 
  %[x,IterN,realTol,objtrue,time] = ...
  pdco(c,A,b,bl,bu,d1,d2,options,x0,y0,z0,xsize,zsize,lsmr_solver);
  
%dictInfo(x,b,IterN,realTol,objtrue,time,Name);
end

function [A,y]= PDCO_partialDCT( m,n )                               % Normalize cols of A


J = randperm(n); J = J(1:m); % m randomly chosen indices

 % generate the m*n partial DCT matrix whose m rows are
 % the rows of the n*n DCT matrix at the indices specified by J
 % see files at @partialDCT/
 A = partialDCT(n,m,J); % A
 % spiky signal generation
 T = 10; % number of spikes
 x0 = zeros(n,1);
 q = randperm(n);
 x0(q(1:T)) = sign(randn(T,1));

 % noisy observations
 sigma = 0.01; % noise standard deviation
 y = A*x0 + sigma*randn(m,1);
%run the l1-regularized least squares solver
% tic;
% [iter,primobj,gap,x] = l1_ls(A,A',m,n,y,lambda,rel_tol);
% time = toc;
%dictInfo(x,y,iter,gap,primobj,time,Name);
end

function [A,b]=randPDCO(m,n)
%   
  rand('state',0);
  en      = ones(n,1);
  em      = ones(m,1);
%   d2      = em;         % D2 = I
  %d2(m-nc+1:m) = delta; % Satisfy last nc rows more accurately.
  
  density = 0.25;
  rc      = 1e-1;
  A       = opMatrix(sprand(m,n,density,rc));
%   
%   x       = en;
%   b       = full(A*x);% + d2;  % Plausible b, as if dual vector y = ones(m,1).
 T = (m+n)/10; % number of spikes
 x0 = zeros(n,1);
 q = randperm(n);
 sigma = 0.01; 
 x0(q(1:T)) = sign(randn(T,1));
b = A*x0 + sigma*randn(m,1);
% b-A*ones(n,1);

  
end