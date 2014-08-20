function [nnz,nnzv,v] = sparsity(v,tol)    
 xmax = tol*norm(v,Inf); 
 nnzv = size(find((abs(v)>xmax)>0));
 nnz = max(nnzv)/max(size(v))*100;
end