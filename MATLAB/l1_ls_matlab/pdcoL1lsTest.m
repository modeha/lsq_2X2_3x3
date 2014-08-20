 function pdcoL1lsTest()
 clc
 clear all
 clc
  m = 340;
  n = 350;
  Q = rand(m,n);
  y = rand(m,1);
  %pdcotestBPDN()
  %PDCOTest( m,n );
%  [x,IterN,realTol,objtrue,time]=pdcorandom(dct(Q),y);
 partialdct(m,n)
 %printResults()
 end
function partialdct(m,n)
% generate the m*n partial DCT matrix whose m rows are
 % the rows of the n*n DCT matrix at the indices specified by J
 % see files at @partialDCT/
%  m = 128;
%  n = 1024;
for i=1:1
    m = i*m;
    n = i*n;
 Name = m;
 nc = m;
 % spiky signal generation
 T = 10*i; % number of spikes
 x0 = zeros(n,1);
 q = randperm(n);
 x0(q(1:T)) = sign(randn(T,1));

 % noisy observations
 sigma = 0.01; % noise standard deviation
 %x0 = sprandvec(m,sparse);
 A = dct(rand(m,n));
 y = A*x0 + sigma*randn(m,1);
 

 lambda_max = 1e-3; % regularization parameter
 rel_tol = 1e-4; % relative target duality gap

%run the l1-regularized least squares solver
% tic;
% [iter,primobj,gap,x] = l1_ls(A,y,lambda_max,rel_tol);
% time = toc;
% type = 'l1ls';
% dictInfo(x,y,iter,gap,primobj,time,Name,type);

  options = pdcoSet;

  x0    = 0.5*ones(n,1);     % Initial x
  y0    = zeros(m,1);        % Initial y
  z0    = ones(n,1);         % Initial z
  xsize = 1;                 % Estimate of norm(x,inf) at solution
  zsize = 1;                 % Estimate of norm(z,inf) at solution

  options.mu0       = 1e-0;  % An absolute value
  options.LSMRatol1 = 1e-4;  % For LPs, LSQR must solve quite accurately
  options.wait      = 0;     % Allow options to be reviewed before solve
  options.Method    = 3;  % Will change to 1 or 3
 
  
  c       = zeros(n,1);
  b       = y;
  em      = ones(m,1);
  en      = ones(n,1);
  zn      = zeros(n,1);
  bigbnd  = 1e+30;
  bl      = -bigbnd*en;         % x > 0
  bu      = bigbnd*en;%10*en;      % x < 10
%   bl(1)   = - bigbnd;   % Test "free variable" (no bounds)
%   bu(1)   =   bigbnd;

  gamma   = 1e-3;       % Primal regularization.
  delta   = 1;       % 1e-3 or 1e-4 for LP;  1 for Least squares.
  d1      = gamma;      % Scalar for this test.
  d2      = em;         % D2 = I
  d2(m-nc+1:m) = delta; % Satisfy last nc rows more accurately.
  
%   rand('state',0);
%   density = 0.25;
%   rc      = 1e-1;
  %A       = sprand(m,n,density,rc);
  %x       = en;
  %b       = full(A*x) + d2;  % Plausible b, as if dual vector y = ones(m,1).
  %c       = rand(n,1);
   options
 type = 'pdco';
  %operator  =  isa(A,'partialDCT')
  [x,IterN,realTol,objtrue,time] = ...
    pdco(c,A,b,bl,bu,d1,d2,options,x0,y0,z0,xsize,zsize);
    dictInfo(x,b,IterN,realTol,objtrue,time,Name,type);
end
end

% keyboard
function [x,IterN,realTol,objtrue,time,Name,b]= PDCOTest( m,n ) % Private function below
% D  = sum(A,1);   D(find(D==0)) = 1;
% D  = sparse( 1:n, 1:n, 1./D, n, n );
% A  = A*D;                                % Normalize cols of A



%rand('state',0);randn('state',0); %initialize (for reproducibility)

%  n = 1024; % signal dimension
%  m = 128; % number of measurements
%  nc = 10;
nc = m;
m2n = [m,n];
x = textscan(num2str(m2n),'%s');
Name = strcat(char(x{1}{1}),char('101'),char(x{1}{2}));
Name = str2num(Name)


 J = randperm(n); J = J(1:m); % m randomly chosen indices

 % generate the m*n partial DCT matrix whose m rows are
 % the rows of the n*n DCT matrix at the indices specified by J
 % see files at @partialDCT/
 A = partialDCT(n,m,J); % A
 At = A'; % transpose of A

 % spiky signal generation
 T = 10; % number of spikes
 x0 = zeros(n,1);
 q = randperm(n);
 x0(q(1:T)) = sign(randn(T,1));

 % noisy observations
 sigma = 0.01; % noise standard deviation
 y = A*x0 + sigma*randn(m,1);

 lambda = 0.01; % regularization parameter
 rel_tol = 0.0001; % relative target duality gap

%run the l1-regularized least squares solver
% tic;
% [iter,primobj,gap,x] = l1_ls(A,At,m,n,y,lambda,rel_tol);
% time = toc;
%dictInfo(x,y,iter,gap,primobj,time,Name);
 
 
 
  options = pdcoSet;

  x0    = 0.5*ones(n,1);     % Initial x
  y0    = zeros(m,1);        % Initial y
  z0    = ones(n,1);         % Initial z
  xsize = 1;                 % Estimate of norm(x,inf) at solution
  zsize = 1;                 % Estimate of norm(z,inf) at solution

  options.mu0       = 1e-0;  % An absolute value
  options.LSMRatol1 = 1e-4;  % For LPs, LSQR must solve quite accurately
  options.wait      = 0;     % Allow options to be reviewed before solve
  options.Method    = 3;  % Will change to 1 or 3
                          %1=Cholesky  2=QR  3=LSMR  4=MINRES  21=SQD

  
  c       = zeros(n,1);
  b       = y;
  em      = ones(m,1);
  en      = ones(n,1);
  zn      = zeros(n,1);
  bigbnd  = 1e+30;
  bl      = zn;         % x > 0
  bu      = bigbnd;%10*en;      % x < 10
%   bl(1)   = - bigbnd;   % Test "free variable" (no bounds)
%   bu(1)   =   bigbnd;

  gamma   = 1e-3;       % Primal regularization.
  delta   = 1e-3;       % 1e-3 or 1e-4 for LP;  1 for Least squares.
  d1      = gamma;      % Scalar for this test.
  d2      = em;         % D2 = I
  d2(m-nc+1:m) = delta; % Satisfy last nc rows more accurately.
  
%   rand('state',0);
%   density = 0.25;
%   rc      = 1e-1;
  %A       = sprand(m,n,density,rc);
  %x       = en;
  %b       = full(A*x) + d2;  % Plausible b, as if dual vector y = ones(m,1).
  %c       = rand(n,1);
   options
 
  %operator  =  isa(A,'partialDCT')
%   [x,IterN,realTol,objtrue,time] = ...
    pdco(c,A,b,bl,bu,d1,d2,options,x0,y0,z0,xsize,zsize);
%dictInfo(x,b,IterN,realTol,objtrue,time,Name);
end

function dictInfo(x,x0,IterN,realTol,objtrue,time,name,type)
if strcmp(type ,'l1ls')
    if exist('l1ls.mat','file')
        load('l1ls','l1ls');
    else
        l1ls = java.util.Hashtable;
    end
    nnzminimizer = sparsity(x,1e-4);
    nnzoriginal  = sparsity(x0,1e-4);
    
    
    t=l1ls.get('Number of iteration');
    d = size(t);
    t(d+1) = IterN;
    l1ls.put('Number of iteration', t);
    
    
    t = l1ls.get('Name of problem');
    d = size (t);
    t(d+1) = name;
    l1ls.put('Name of problem',t);
    
    
    t = l1ls.get('Time');
    d = size (t);
    t(d+1) = time;
    l1ls.put('Time',t);
    
    t = l1ls.get('Objective');
    d = size (t);
    t(d+1) = objtrue;
    l1ls.put('Objective',t);
    
    t = l1ls.get('Number of non zero in minimizer');
    d = size (t);
    t(d+1) = nnzminimizer;
    l1ls.put('Number of non zero in minimizer',t);
    
    t = l1ls.get('Number of non zero in original');
    d = size (t);
    t(d+1) = nnzoriginal;
    l1ls.put('Number of non zero in original',t);
    
    t = l1ls.get('complementary');
    d = size (t);
    t(d+1) = realTol;
    l1ls.put('complementary',t);
 
    save('l1ls','l1ls');
else
    if exist('pdco.mat','file')
        load('pdco','pdco');
    else
        pdco = java.util.Hashtable;
    end
    
    nnzminimizer = sparsity(x,1e-4);
    nnzoriginal  = sparsity(x0,1e-4);
    
    t=pdco.get('Number of iteration');
    d = size(t);
    t(d+1) = IterN;
    pdco.put('Number of iteration', t);
    
    
    t = pdco.get('Name of problem');
    d = size (t);
    t(d+1) = 1;
    pdco.put('Name of problem',t);
    
    
    t = pdco.get('Time');
    d = size (t);
    t(d+1) = time;
    pdco.put('Time',t);
    
    t = pdco.get('Objective');
    d = size (t);
    t(d+1) = objtrue;
    pdco.put('Objective',t);
    
    t = pdco.get('Number of non zero in minimizer');
    d = size (t);
    t(d+1) = nnzminimizer;
    pdco.put('Number of non zero in minimizer',t);
    
    t = pdco.get('Number of non zero in original');
    d = size (t);
    t(d+1) = nnzoriginal;
    pdco.put('Number of non zero in original',t);
    
    t = pdco.get('complementary');
    d = size (t);
    t(d+1) = realTol;
    pdco.put('complementary',t);
    
%     t = pdco.get('primalinfeas');
%     d = size (t);
%     t(d+1) = Pinf;
%     pdco.put('primalinfeas',t);
%     
%     t = pdco.get('dualinfeas');
%     d = size (t);
%     t(d+1) = Dinf;
    pdco.put('gap',t);
    save('pdco','pdco');
end
end

function printResults()
load('pdco','pdco');
    Number_of_Iteration_pdco = pdco.get('Number of iteration');
    Name_pdco = pdco.get('Name of problem');
    Time_pdco = pdco.get('Time');
    Objective_pdco = pdco.get('Objective');
    nnzmnimizer_pdco = pdco.get('Number of non zero in minimizer');
    nnzoriginal_pdco = pdco.get('Number of non zero in original');
    gap_pdco =pdco.get('gap');
    
load('l1ls','l1ls');

    Number_of_Iteration_l1ls = l1ls.get('Number of iteration');
    Name_l1ls  = l1ls.get('Name of problem');
    Time_l1ls = l1ls.get('Time');
    Objective_l1ls = l1ls.get('Objective');
    nnzmnimizer_l1ls = l1ls.get('Number of non zero in minimizer');
    nnzoriginal_l1ls = l1ls.get('Number of non zero in original');
    gap_l1ls = l1ls.get('complementary');
fprintf('name iter \t obj \t\t gap \t\t time \t\t nnz_original \t nnz_minimizer\n');
dim = size(Time_pdco);
for i=1:dim(1)

fprintf('%-4d %-4d\t%-.6e\t%-.6e\t%-.6e\t%-.6e\t %-.6e %5s\n',...
    Name_pdco(i),Number_of_Iteration_pdco(i),gap_pdco(i),Objective_pdco(i),...
    Time_pdco(i), nnzoriginal_pdco(i), nnzmnimizer_pdco(i),'   PDCO');
fprintf('\n');
fprintf('%-4d %-4d\t%-.6e\t%-.6e\t%-.6e\t%-.6e\t %-.6e %5s \n',...
    Name_l1ls(i),Number_of_Iteration_l1ls(i),gap_l1ls(i),Objective_l1ls(i),...
    Time_l1ls(i), nnzoriginal_l1ls(i), nnzmnimizer_l1ls(i),'   l1ls');
fprintf('\n');
end
end

%en=pdco.keys();

% while (en.hasMoreElements())
%     java.lang.System.out.println(en.nextElement())
%     pdco.get(en.nextElement())
% end
% keyboard
function pdcotestBPDN()% m,n,k,lambda )
clc

 m=15;  n=30;  k = 4;  lambda=1e-3; % pdcotestBPDN( m,n,k,lambda );
% Generates a random m by n basis pursuit denoising problem (BPDN)
% with k non zeros in the optimal solution, and treats the constraint matrix
% A as an operator.  (We need k <= n.)
%
% BPDN is the problem
%    minimize lambda ||x||_1 + 1/2 ||Ax-b||^2
% where A is m by n with m<n.
%
% The problem is tranfsormed into 
%
%    minimize    lambda e'[xl;xh] + 1/2 ||D1*[xl;xh]||^2 + 1/2 ||r||^2
%      xl,xh,r
%    subject to  [A -A]*[xl;xh] + r = b,   0 <= x <= inf,   r unconstrained,
%
% where A is m by n, e is a 2n-vector of ones,
% D1 is a small positive-definite diagonal matrix,
% and everything relevant is the same as in pdco.m.
% Here, D1 is defined by d1 in the private function toydata.

%-----------------------------------------------------------------------
% 09 Apr 2012: Example BPDN test program for pdco.m,
%              derived from pdcotestLS.m and inspired by
%              Michael Friedlander and Ewout van den Berg's spgl1 examples.
%              Santiago Akle and Michael Saunders, ICME, Stanford University.
%-----------------------------------------------------------------------

  if nargin < 4
     lambda = 1e-3;
  end

  [A,b,bl,bu,c,d1,d2,J,x_Sol] = toydata( m,n,k,lambda ); % Private function below
  
  options = pdcoSet;

  x0    = 0.5*ones(2*n,1);   % Initial x
  y0    = zeros(m,1);        % Initial y
  z0    = ones(2*n,1);       % Initial z
  xsize = 1;                 % Estimate of norm(x,inf) at solution
  zsize = 1;                 % Estimate of norm(z,inf) at solution

  options.mu0       = 1e-0;  % An absolute value
  options.LSMRatol1 = 1e-9;  % LSMR and MINRES must solve quite accurately
  options.Method    = 3; 
  options.wait      = 0;   % Allow options to be reviewed before solve
  
  [x,IterN,realTol,objtrue,time] = ...
    pdco(c,A,b,bl,bu,d1,d2,options,x0,y0,z0,xsize,zsize);
  
% Rebuild the solution
%   xSparse  = x(1:n) - x(n+1:end);
%   figure
%   hold on
%   stem(xSparse,'rx');
%   stem(x_Sol  ,'bo');
%   hold off

% keyboard               % Allow review of x,y,z, etc.
%-----------------------------------------------------------------------
% end function pdcotestBPDN
%-----------------------------------------------------------------------
end

function [Aop,b,bl,bu,c,d1,d2,J,xSol] = toydata( m,n,k,lambda )

%        [Aop,b,bl,bu,c,d1,d2] = toydata( m,n,k,lambda );
% defines an m by n matrix A, rhs vector b, and scalar lambda
% for the BPDN problem
%
%        minimize lambda ||x||_1 + 1/2 ||Ax-b||^2,
%
% where k is the expected number of nonzeros in x.

%-----------------------------------------------------------------------
% 10 Apr 2012: Adapted from version in pdcotestLS.m 
%-----------------------------------------------------------------------

A =dct(randn(n,m))';


  p       = randperm(n);      % Pick a random support  
  xSol    = zeros(n,1);       % Form a sparse solution
  J       = p(1:k);
  xSol(J) = randn(k,1);
  b       = A*xSol;           % Form the corresponding rhs

  % Put the data into the format that PDCO understands
  Aop     = @(mode,m,n,x) operatorA(mode,x,A);
  c       =  ones(2*n,1)*lambda;
  bl      = zeros(2*n,1);     % x > 0
  bu      =   inf(2*n,1);
  d1      = 1e-10;            % Primal regularization
  d2      = 1;                % Make it a regularized least squares problem
%-----------------------------------------------------------------------
% end private function toydata
%-----------------------------------------------------------------------

end
function y = operatorA( mode,x,A )

% Computes products with [A -A] and [A -A]' to be used
% by pdco as an operator.

   [m,n] = size(A);
   if mode == 1
      y = A*(x(1:n) - x(n+1:2*n))
   else
      r = A'*x;
      y = [r;-r];
   end
end

%----------------------------------------------------------------------
% End private function operatorA
%----------------------------------------------------------------------


function [x,IterN,realTol,objtrue,time]= pdcorandom(Q,y)
options = pdcoSet;

 [m,n] = size(Q);
 M = m+2*n;
 N = 4*n;
 
 
 delta = 0.01;
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
  
  %     ucon = np.zeros(2*n)
%     lcon = -np.ones(2*n)*inf
%     uvar = np.ones(2*n)*inf
%     lvar = -np.ones(2*n)*inf
% 
%  DD = zeros(3*n);
%  DD(1:n,1:n)         = I;
%  DD(n+1:2*n,n+1:2*n) = I;
%  DD(2*n+1:3*n,2*n+1:3*n) = I;
 
  em      = ones(M,1);
  en      = ones(N,1);
  zn      = zeros(n,1);
  on      = ones(n,1);
  bigbnd  = 1e+30;
  inf = bigbnd*on;
  
  
  bl      = [-inf;zn;zn;-inf];         % x > 0
  bu      = [inf;inf;inf;inf];%10*en;      % x < 10
%   bl(1)   = - bigbnd;   % Test "free variable" (no bounds)
%   bu(1)   =   bigbnd;

  gamma   = 1e-3;       % Primal regularization.
  delta   = 1e-3;       % 1e-3 or 1e-4 for LP;  1 for Least squares.
  d1      = gamma;      % Scalar for this test.
  d2      = em;         % D2 = I
  nc = N/10;
  d2(M-nc+1:M) = delta; % Satisfy last nc rows more accurately.
[x,IterN,realTol,objtrue,time] = ...
    pdco(c,A,b,bl,bu,d1,d2,options,x0,y0,z0,xsize,zsize);
end