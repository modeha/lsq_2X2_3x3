
import numpy as np
import time

from pysparse import spmatrix
from dctt import dctt, test, idctt,l1_ls_itre
from pysparse.sparse.pysparseMatrix import PysparseMatrix 
#from pylab import figure,subplot,plot,grid,title,ylabel,semilogy,xlabel,show

from pykrylov.lls import LSQRFramework
from pykrylov.lls import LSMRFramework
from pykrylov.lls import CRAIGFramework
from pykrylov.linop import PysparseLinearOperator
from pykrylov.linop import BaseLinearOperator,DiagonalOperator
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
from toolslsq import as_llmat

from pysparse.sparse.pysparseMatrix import PysparseMatrix,\
     PysparseIdentityMatrix, PysparseSpDiagsMatrix
from numpy import ones
import scipy.io as sc

def datas(A=None,B=None,C=None,rhs=None):
    """Creates a ll_mat object from a file named fileName,
    which must be in MatrixMarket Coordinate format.
    Depending on the file content, either a symmetric or 
    a general sparse matrix is generated."""
    A = np.load(A)
    A = A[A.files[0]]

    B = np.load(B)
    B = B[B.files[0]]

    C = np.load(C)
    C = C[C.files[0]]

    rhs = np.load(rhs)
    rhs = rhs[rhs.files[0]]

    #m, n = A.shape
    #e = ones(m)
    #I = PysparseSpDiagsMatrix(size=m, vals=(0*e,-e), pos=(-1,0))
    #A = spmatrix.matrixmultiply(as_llmat(A),as_llmat(I))
    return A, B, C,rhs
def solve_iterative_opr(A,B,C,rhs):
    A= sp(matrix=as_llmat(A))
    A = PysparseLinearOperator(A)

    B= sp(matrix=as_llmat(B))
    #print B
    B = PysparseLinearOperator(B)

    C= sp(matrix=as_llmat(C))
    C = PysparseLinearOperator(C)
    
    m,n = B.shape 
    p, q = A.shape
    e = ones(p)
    M = DiagonalOperator(1/(A*e))
    mc,nc = C.shape
    ec = ones(mc)
    N = DiagonalOperator(1/(C*ec))
    #Construisons g,f et resolvons Ax_0=f, b=[g-Bx,0]
    g = rhs[n:]
    f = rhs[:n] 
    x_0 = -M*f
    b = g-B*x_0
    lsqr = LSQRFramework(B)
    #lsqr = CRAIGFramework(B)
    #lsqr = LSMRFramework(B)
    t0 = time.clock()
    lsqr.solve(b,atol = 1e-16, btol = 1e-16,M = N, N = M,show = True,\
                 store_resids = True)
        #lsqr.solve(b,atol = 1e-16, btol = 1e-16,M = N, N = M)
    t1 = time.clock() 
    print "CPU TIME:",t1-t0
    print lsqr.r2norm,lsqr.r1norm
#Reconstruction de le solution, calcul des erreurs
    xsol = x_0 + lsqr.x
    w = b - B * lsqr.x
    ysol = N(w)
    SolFinal = np.concatenate((xsol, ysol), axis=0)
    

def solve_iterative(A,B,C,rhs):
    ""
    m,n = B.shape 

    B =  sp(matrix=as_llmat(B))

    A = sp(matrix=as_llmat(A))
    B = PysparseLinearOperator(B)
    C = sp(matrix=as_llmat(C))

    #x_M = bSol[:n]
    #y_M = bSol[n:]
    
    M=DiagonalOperator(1/A.takeDiagonal())
    N=DiagonalOperator(1/C.takeDiagonal())

#Construisons g,f et resolvons Ax_0=f, b=[g-Bx,0]
  
    g = rhs[n:]
    f = rhs[:n] 

    x_0 = -M*f
   
    b = g-B*x_0

    #lsqr = LSQRFramework(B)
    lsqr = LSMRFramework(B)
    #lsqr = CRAIGFramework(B)
    

    tp1 = time.clock()
    lsqr.solve(b,atol = 1e-16, btol = 1e-16,M = N, N = M)
    # lsqr.solve(b,atol = 1e-16, btol = 1e-16,M = N, N = M,show = True,\
    #            store_resids = True)
    tp2 = time.clock() 
    # print "CPU TIME:",tp2-tp1
    # print lsqr.r2norm,lsqr.r1norm,


#Reconstruction de le solution, calcul des erreurs

    xsol = x_0 + lsqr.x
    w = b - B * lsqr.x
    ysol = N(w)

    SolFinal = np.concatenate((xsol, ysol), axis=0)
    return sol_final
   
        
# def norm2(A,x):
#     return np.dot(x, A*x)

if __name__ == '__main__':
    dirs = '/Users/Mohsen/Documents/nlpy_mohsen/lsq/tp_npz/'
    #(78,) (36, 36) (42, 36) (42, 42)
    # A,y = l1_ls_itre(36,36)
    # B,y = l1_ls_itre(42,36)
    # C,y = l1_ls_itre(42,42)
    # Q,rhs = l1_ls_itre(78,78)
    # rhs = rhs[:,0]
    # A = A.to_array()
    # B = B.to_array()
    # C = C.to_array()
    #print rhs[:,0].shape,A.shape,B.shape,C.shape


    A, B, C, rhs = datas(A=dirs+'A.npz',B=dirs+'B.npz',C=dirs+'C.npz',rhs=dirs+'rhs.npz')
    #solve_iterative(A,B,C,rhs)
    solve_iterative_opr(A,B,C,rhs)
