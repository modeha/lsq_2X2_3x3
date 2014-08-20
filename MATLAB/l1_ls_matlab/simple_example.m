function simple_example(~,varargin)
params.j = 1;
params.lambda_max = 0;
params.read_from_file = 0;
params.sample = 0;
param.testproblems = 0;

%if params.sample 
%   simple example to show the usage of l1_ls
% problem data
% A  = [1    0    0   0.5;
%       0    1  0.2   0.3;
%       0  0.1    1   0.2];
%  A = dct2(rand(60))
%      % 1    0    1     1];
% x0 = [11 0 1 0]';    % original signal
% x0 = rand(1,60)';
% y  = A*x0         % measurements with no noise
% [x,status]=l1_ls(A,y,lambda,rel_tol);
% norm(x-x0)
% x
% x0



m =dim(i,1);
n =dim(i,2);
mn=[m,n];
x = strread(num2str(mn),'%s');
name=strcat(char(x(1)),char('_'),char(x(2)));
root = strcat('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/',read_from_file,'/');
if params.read_from_file ~='0'
    A = importdata(strcat(root,'/Q.txt'));
    y = importdata(strcat(root,'/d.txt'));
    x0 = importdata(strcat(root,'/x0.txt'))
else
    A = dct2(rand(m,n));
    y0 = rand(n,1);% cutoff point or y0 = -n:n;  original signal
    y = A*y0;
end


A_1 = sqrt(2)/2*A;
y_1 = sqrt(2)/2*y;

if params.lambda_max ==0
    [lambda_max] = max(1e-2,1/find_lambdamax_l1_ls(A',y));
end
diary ON;
tic
%[iter,primobj,gap] = l1_ls(A_1,y_1,lambda_max);
%[iter,primobj,gap] = l1_ls_direct(A_1,y_1,lambda_max);
time = toc;

diary(strcat(char('results.txt')));
lambda = lambda_max
fid = fopen('/Users/Mohsen/Documents/nlpy_mohsen/lsq/matlab.txt','a+');
defferent = x-x0;
m2n = [m,2*n];
x = strread(num2str(m2n),'%s');
Name = strcat(char(x(1)),char('_'),char(x(2)),char('_'),char(x(2)));
fprintf( fid,'''%s''\t\t :[%d,%.6e,%.6e,%.6e], \n',Name,iter,primobj,gap, time);
fprintf('name \t iter \t obj \t\t gap \t\t time \t\t nnz_original \t nnz_minimizer \t norm(x_x0)\n');
fprintf('%-10s%-4d\t%-.6e\t%-.6e\t%-.6e\t %-.6e\t %-.6e \t%-.6e \n',Name,iter,primobj,gap, time, nnz0,nnzf,norm(defferent));
fclose(fid);
diary OFF;
cd('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB');
mkdir(name)
save -ascii '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/Q.txt' A
save -ascii '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/d.txt' y 
save -ascii '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/delta.txt' lambda
copyfile('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/Q.txt',name)
copyfile('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/d.txt',name)
copyfile('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/delta.txt',name)
copyfile('/Users/Mohsen/Documents/MATLAB/results.txt',name)
delete('/Users/Mohsen/Documents/MATLAB/results.txt')
cd ('/Users/Mohsen/Documents/MATLAB/');
clear all
end