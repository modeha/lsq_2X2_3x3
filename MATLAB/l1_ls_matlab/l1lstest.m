clc
Q  = [1    0    0   0.5;
    0    1  0.2   0.3;
    0  0.1    1   0.2];
m = 500; n = 1400; sparse = 5;
x0 = [1 0 1 0]';  % original signal
%x0 = sprandvec(m,sparse);
%x0 = rand(n,1);
%A = gallery('binomial',m);
A = hilbert(m,n);
Q=dctt(A);
%Q = dct2(rand(n,m));
y = sum(Q')';%+0.01*randn(n,1);
%y = Q*x0;
lambda_max = 1e-3;%max(1e-2,1/find_lambdamax_l1_ls(Q',y))
[m,n] = size(Q);
%lambda = 0.01
A_1 = sqrt(2)/2*Q;
y_1 = sqrt(2)/2*y;
l1_ls(A_1,y_1,lambda_max);
l1_ls_direct(A_1,y_1,lambda_max);

