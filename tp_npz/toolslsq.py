#! /usr/bin/python2.7
# -*-coding:Utf-8 -*

import numpy as np
#from nlpy.model.nlp import NLPModel 
from pysparse import spmatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix
#from cvxopt import normal, sparse
#import cvxopt

__docformat__ = 'restructuredtext'

def as_llmat(A,tol= 1e-10):
    """Fill out any matrix as ll_mat type."""
    if isinstance(A,np.ndarray):
        dim = A.shape
        m,n = dim[0],dim[1]
        if min(m,n)>1:
            spmat_ = spmatrix.ll_mat(m,n) 
            nz = np.nonzero(A)
            r = nz[0].size
            for i in range(r):
                ii = np.int(nz[0][i])
                ij = np.int(nz[1][i])
                spmat_[ii,ij] = A[nz[0][i],nz[1][i]]
            spmat= spmat_
            return spmat
        else:
            n,m =dim
            vec = np.zeros((1,max(n,m)))
            for i in range(max(n,m)):
                vec[0,i] = A[0,i]
            return vec
    elif isinstance(A, PysparseMatrix):
        dim = A.shape
    
        m,n =dim
        spmat_ = spmatrix.ll_mat(m,n) 
        #nz = np.nonzero(A)
        #print nz
 
        for i in range(m):
            for j in range(n):
                spmat_[i,j] = A[i,j]
        spmat= spmat_
        return spmat
    else:
        t = type(A)
        m,n= A.shape
        spmat_ = spmatrix.ll_mat(m,n) 
        spmat_[:,:]=A
        return spmat_
        
    

def contiguous_array(x):
    "Return a contiguous array in memory (C order)."
    return np.ascontiguousarray(x, dtype=float)


    
if __name__ == '__main__':
    (m, n, p) = (4, 5, 6)  
    x= np.random.rand(n)
    Q= np.random.rand(p,n)

    Q=to_llmat(Q)
    Q = PysparseMatrix(matrix=Q)
    print Q*x
    Q2 = to_llmat(Q)
   
    to_llmat(PysparseMatrix(matrix=Q2))
    print isinstance(Q2, PysparseMatrix)
