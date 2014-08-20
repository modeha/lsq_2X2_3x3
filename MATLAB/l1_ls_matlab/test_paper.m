% problem data
lambda = 0.01;      % regularization parameter
rel_tol = 0.01;     % relative target duality gap

A  = [1    0    0   0.5;
      0    1  0.2   0.3;
      0  0.1    1   0.2;
      1  0      1    1];
 %A = dct2(rand(60))
     % 1    0    1     1];
x0 = [1 0 1 0]';    % original signal
%x0 = rand(1,60)';
y  = A*x0         % measurements with no noise
[x,status]=l1_ls(A,y,lambda,rel_tol);
norm(x-x0)
x
save -ascii '/Users/Mohsen/Documents/MATLAB/matpydata/Q.txt' A
save -ascii '/Users/Mohsen/Documents/MATLAB/matpydata/d.txt' y 
save -ascii '/Users/Mohsen/Documents/MATLAB/matpydata/delta.txt' lambda