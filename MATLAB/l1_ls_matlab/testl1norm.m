 % Matlab script for solving the sparse signal recovery problem
 % using the object-oriented programming feature of Matlab.
 % The three m files in ./@partialDCT/ implement the partial DCT class
 % with the multiplication and transpose operators overloaded.

 rand('state',0);randn('state',0); %initialize (for reproducibility)

 n = 1024; % signal dimension
 m = 128; % number of measurements

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
 rel_tol = 0.01; % relative target duality gap

%run the l1-regularized least squares solver

 [x,status]=l1_ls(A,At,m,n,y,lambda,rel_tol);
 
 
 A=[
    0.2867    0.8133    0.2936
    0.9721    0.7958    0.3033
    0.6198    0.6210    0.1496
    0.2760    0.8500    0.6658];
idct(A);