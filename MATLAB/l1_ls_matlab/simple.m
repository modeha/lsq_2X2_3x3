function [iter, primobj, gap, time, nnz0, nnzf, Name] = simple(~,varargin)
params.n = 0;
params.m = 0;
params.density = 0.4;
params.lambda_max = 0;
params.sample = 0;
params.testproblems = 0;
params.sparse = 0;
params.type = 0;
params.Solver_Type = 'InDirect';
params.Sparsity = 'False';
params.Tol = 1e-4;
params = parse_pv_pairs(params,varargin)


different = 0;
density = params.density;
lambda_max = params.lambda_max ;
sparse = params.sparse ;
type = params.type ;
testproblems = params.testproblems;
Solver_Type = params.Solver_Type;
Sparsity = params.Sparsity;
tol = params.Tol;
n = max(params.m,params.n) ;
m = min(params.m,params.n) ;

[Q,x0,y,Type_of_Problem] = peremeters(type,testproblems,density,sparse,m,n);

fprintf('Type of Problem : %s\n',Type_of_Problem)

if lambda_max == 0
    lambda_max = max(1e-2,1/find_lambdamax_l1_ls(Q',y));
end

mn =size(Q);
x = textscan(num2str(mn),'%s');
A_1 = sqrt(2)/2*Q;
y_1 = sqrt(2)/2*y;
diary ON;
fprintf('lambda_max: %f condition Number of A: %f\n',lambda_max,cond(A_1))
tic 
 if strcmp(Solver_Type ,'Direct')
    [iter,primobj,gap,x] = l1_ls_direct(A_1,y_1,lambda_max);
 elseif strcmp(Solver_Type ,'InDirect')
    [iter,primobj,gap,x] = l1_ls(Q,y,lambda_max,1e-2);
 end

time = toc;
if strcmp(Sparsity,'Ture')
    [nnz0] = sparsity(x0,tol);
    [nnzf] = sparsity(x0,tol);
    different = x-x0;
elseif strcmp(Sparsity,'False')
    nnz0=0;nnzf=0;
end
diary(strcat(char('results.txt')));
lambda = lambda_max;
paths = strcat('/Users/Mohsen/Documents/nlpy_mohsen/lsq/matlab','.txt');

fid = fopen(char(paths),'a+');
%fid = fopen('/Users/Mohsen/Documents/nlpy_mohsen/lsq/matlab.txt','a+');
%
m2n = [mn(2)*2,mn(1),mn(1)];
x = textscan(num2str(m2n),'%s');
Name = strcat(char(x{1}{1}),char('_'),char(x{1}{2}),char('_'),char(x{1}{3}));
fprintf('Dimention of Problem: %s\n\n',Name)
name = Name;
%Name = strcat(char(x(1)),char('_'),char(x(2)),char('_'),char(x(2)),char('_'),char(sparse));
fprintf( fid,'''%s''\t\t :[%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e], \n',Name,iter,primobj,gap, time,nnz0,nnzf,norm(different));

fprintf('name \t\titer \t\t obj \t\t gap \t\t time \t\t nnz_original \t nnz_minimizer \t norm(x_x0)\n');
fprintf('%-10s \t%-4d\t\t%-.6e\t%-.6e\t%-.6e\t %-.6e\t %-.6e \t%-.6e \n',Name,iter,primobj,gap, time, nnz0,nnzf,norm(different));

fclose(fid);
diary OFF;
cd('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB');
mkdir(name)
save -ascii '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/Q.txt' Q
save -ascii '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/x0.txt' x0
save -ascii '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/d.txt' y
save -ascii '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/delta.txt' lambda
copyfile('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/Q.txt',name)
copyfile('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/d.txt',name)
copyfile('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/x0.txt',name)
copyfile('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/delta.txt',name)
copyfile('/Users/Mohsen/Documents/MATLAB/results.txt',name)
delete('/Users/Mohsen/Documents/MATLAB/results.txt')
cd ('/Users/Mohsen/Documents/MATLAB/');
clear all
end


function [Q,x0,y,name] = peremeters(type,testproblems,density,sparse,m,n)

if strcmp(type, 'sample')
    %Simple example to show the usage of l1_ls problem data
    Q  = [1    0    0   0.5;
          0    1  0.2   0.3;
          0  0.1    1   0.2;
          1    0    1   1]
    x0 = [1 0 1 0]';    % original signal
    y  = Q*x0;         % measurements with no noise
    name = 'sample';
    
elseif strcmp(type ,'read_from_file')
    root = strcat('/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/',char(testproblems));
    strcat(root,'/Q.txt');
    Q = importdata(strcat(root,'/Q.txt'));
    x0 = importdata(strcat(root,'/x0.txt'));
    y = importdata(strcat(root,'/d.txt'));
    name = 'Saved_Problem';
    
elseif strcmp(type ,'testproblems')
    dim=[8,6;
        32,4;
        64,8;
        16,12;
        128,16;
        256,32;
        304,88;
        512,64;
        536,192;
        1024,128;
        2048,256;
        4096,512;
        1072,384;
        1576,317;
        2144,768;
        2288,536;
        3192,1024;
        3768,1096;
        4152,1144;
        5384,2048];
    i = dim(k,1);
    j = dim(k,2);
    Q = dct2(rand(i,j));
    x0 = rand(j,1);% cutoff point or x0 = -n:n;  original signal
    y = Q*x0;
    name = 'TestProblem';
    
elseif  strcmp(type,'gauss')
    [x0,Q]=gaussian(n,m);
    y = Q*x0;
    name = 'gauss';
    
elseif strcmp(type,'sparse')
    x0 = sprandvec(m,sparse);
    Q = rand(n,m);
    y = Q*x0;
    name = 'sparse';
    
elseif  strcmp(type,'GaussS')
    x0 = sprandvec(m,sparse);
    Q = dct2(rand(n,m));
    y = Q*x0;%+0.01*randn(n,1);
    name = 'GaussS';
    
elseif strcmp(type ,'sprand')
    x0 = sprandvec(m,sparse);
    Q = sprand(n,m,density);
    y = Q*x0;
    name = 'sprand';
    
elseif strcmp(type, 'direct')
    A = hilbert(n,m);
    Q=dctt(A);
    x0 = sum(Q')';
    y = sum(Q')'+0.01*randn(n,1);
    name = 'direct';
    
else
    error('type error')
end
end