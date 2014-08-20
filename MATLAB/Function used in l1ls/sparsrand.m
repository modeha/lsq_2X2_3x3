%generate a sparse matrix
m =5;
n = 5;
offdiag=sparse(1:m,1:n,2*ones(1,n),m,m);
offdiag2=sparse(4:m, 1:n-2,3*ones(1,n-2),m,m);
offdiag3=sparse(n-4:m, 1:6,7*ones(1,6),m,m);
a=sparse(1:m,1:m,4*ones(1,m),m,m);
a=a+offdiag+offdiag'+offdiag2+offdiag2'+offdiag3+offdiag3';
a=a*a';
%generate full matrix
b=full(a)
% morder=symmmd(a);
% %time & flops
% tic; flops(0);
% spmult=a(morder ,morder)*a(morder,morder)';
% tl=toc; flsp=flops;
% tic; flops(0);
% fulmult=b*b' ;
% t2=toc; flful=flops;
% fprintf('time sparse mult= %4.2f flops sparse mult =%6. 0f\n' ,tl,flsp);
% fprintf('time full mult= %4.2f f lops full mult= %6.0f\n' ,t2,flfu1);