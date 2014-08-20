from lsqmodel import LSQModel, LSQRModel
from dctt import dctt, test, idctt,l1_ls_itre

from numpy import fft, array, arange, zeros, dot, transpose
from pysparse import spmatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
from pysparse.sparse import PysparseIdentityMatrix as eye

from toolslsq import as_llmat, contiguous_array
from pykrylov.tools import check_symmetric
from pykrylov.linop import *
import sys

from math import sqrt, cos, pi
#import Others.copy_past as cp
import numpy as np
from scipy import Infinity as inf
import rogues as rog
import shutil
from shutil import copytree, ignore_patterns
import os,numpy,glob


_folder = str(os.getcwd()) + '/output/tp_npz/'



def remove_type_file(type='.pyc',path= (os.getcwd())):
    
    for file in os.listdir(path):
        if str(file[-len(type):]) ==type:
            os.remove(path+'/'+file)

def first_class_tp(nvar=18, prowQ=9, mcon=4 ):
    """First class of test problems we choose random data 
    generated data such that x=(1,...,1) be the exact solution."""
    n = nvar + prowQ + mcon
    p = prowQ + mcon
    m = mcon
    c = np.zeros(n)
    d = np.zeros(p)
    ucon = np.ones(m)*inf
    uvar = np.ones(n)*inf
    lvar = -np.ones(n)*inf
    #name = str(p)+'_'+str(n)+'_'+str(m)+'_First'+'.txt'
    name = str(p)+'_'+str(n)+'_'+str(m)+'_l1_tp'+'.txt'
    
    # Q randomly chosen such that Qij belong to the (-10,10)
    Q = 10 * np.random.rand(p, n)*(np.random.randint(3, size=(p,n))-1)
    
    # d=(di), di=sum dij for i= 1,...,p
    for i in range(p): 
        d[i]= Q[i,:].sum()
    # B randomly chosen such that Bij belong to the (-3,3)
    B = 3 * np.random.rand(m, n)*(np.random.randint(3, size=(m,n))-1)
    
    b= np.zeros(m)
    b[0] = B[0,:].sum()
    for i in range(m):
            mu = np.random.rand()+ 1e-10
            b[i] = B[i,:].sum()-m*mu
    lcon = b
    return Q,B,d,c,lcon,ucon,lvar,uvar,name 
        
def second_class_tp(p,n):
    """This class  is rather ill conditioned test problems 
    defined by a Hilbert Matrix"""
    c = np.zeros(n)
    d = np.zeros(p)
    ucon = np.zeros(n)
    lcon = np.zeros(n)
    
    #uvar = np.ones(n)*1
    uvar = np.ones(n)*5
    lvar = np.ones(n)*0.5
    name = str(p)+'_'+str(n)+'_'+str(n)+'_l1_tp'+'.txt'
    #name = str(n)+'_'+str(p)+'_'+'_second_tp'+'.txt'
    Q = rog.hilb(p,n)
    # d=(di), di=sum qij for i= 1,...,p
    for i in range(p): 
        d[i]= Q[i,:].sum()
    B = np.zeros((n,n))
    return Q,B,d,c,lcon,ucon,lvar,uvar,name

def third_class_tp(nvar=18, prowQ=9, mcon=4 ):
    "generate randomly a feasible CLSQP."
    n = nvar + prowQ + mcon
    p = prowQ + mcon
    m = mcon
    q = n-mcon
    name = str(p)+'_'+str(n)+'_'+str(m)+'_l1_tp'+'.txt'
    
    
    c =   np.random.randn(n)
    d =   np.random.randn(p) - 2
    Q =   np.random.rand(p, n)*(np.random.randint(3, size=(p,n))-1)
    Bs =  np.random.rand(m, q)*(np.random.randint(3, size=(m,q))-1)

    I = np.identity(m)
    # feasible region is not empty
    B = np.concatenate((Bs,I), axis=1)
    
    u = 1
    l = -1
    
    lcon =  l*np.random.rand(m)
    ucon =  u*np.random.rand(m)
    lvar =  l*np.random.rand(n)
    uvar =  u*np.random.rand(n)

    return Q,B,d,c,lcon,ucon,lvar,uvar,name 

def fourth_class_tp(path=  "/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp/"):
    from pysparse import spmatrix
    from pysparse.sparse.pysparseMatrix import PysparseMatrix
    from  pysparse.sparse import PysparseIdentityMatrix as eye
    Q,_,d,_,_,_,_,_,delta,name =  mat_py(path)
    p,n = Q.shape
    c = np.zeros(n)
    print delta,p,n
    c = np.concatenate((np.zeros(n),np.ones(n)*delta), axis=1)
    ucon = np.zeros(2*n)
    lcon = -np.ones(2*n)*inf
    uvar = np.ones(2*n)*inf
    lvar = -np.ones(2*n)*inf
    
    I = np.identity(n)
    B = np.zeros([2*n, 2*n])
    B[:n,:n] =  I
    B[n:,:n] = -I
    B[:n,n:] = -I
    B[n:,n:] = -I
    Q_ = np.zeros([p,n])
    new_Q = np.append(Q,Q_,axis=1)
    p, n = new_Q.shape
    m, n = B.shape
    name = str(n)+'_'+str(p)+'_'+str(p)+'_l1_tp'+'.txt'
    return new_Q,B,d,c,lcon,ucon,lvar,uvar,name

def fifth_class_tp(p=0,n=0):
    Q,y = test (p,n)
    d = y[:,0]
    #d = np.random.rand(p)
    #Q = dctt(rog.hilb(p,n))
    #for i in range(p):
        #d[i]= Q[i,:].sum()
    c = np.zeros(n)
    delta = 0.01
    c = np.concatenate((np.zeros(n),np.ones(n)*delta), axis=1)
    ucon = np.zeros(2*n)
    lcon = -np.ones(2*n)*inf
    uvar = np.ones(2*n)*inf
    lvar = -np.ones(2*n)*inf

    I = np.identity(n)
    B = np.zeros([2*n, 2*n])
    B[:n,:n] =  I
    B[n:,:n] = -I
    B[:n,n:] = -I
    B[n:,n:] = -I
    Q_ = np.zeros([p,n])
    new_Q = np.append(Q,Q_,axis=1)
    p, n = new_Q.shape
    m, n = B.shape
    name = str(p)+'_'+str(n)+'_'+str(m)+'_l1_ls'
    return new_Q,B,d,c,lcon,ucon,lvar,uvar,name

def sixeth_class_tp(p=0,n=0):
    
    Q,y = test (p,n)
    d = y[:,0]
    
    c = np.zeros(n)
    ucon = np.zeros(n)
    lcon = np.zeros(n)
    
    #uvar = np.ones(n)*1
    uvar = np.ones(n)*10000
    lvar = -np.ones(n)*10000
    name = str(p)+'_'+str(n)+'_l1_ls'
    B = np.zeros((n,n))
    return Q,B,d,c,lcon,ucon,lvar,uvar,name


def nested_remove(L, x):
	if x in L:
	    L.remove(x)
	else:
	    for element in L:
		if type(element) is list:
		    nested_remove(element, x)

def linearoperator():
    np.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)
    Q,B,d,c,lcon,ucon,lvar,uvar,name = fifth_class_tp(8,11)
    #Q,B,d,c,lcon,ucon,lvar,uvar,name = sixeth_class_tp(128,1024)
    lsqpr = lsq_tp_generator(Q,B,d,c,lcon,ucon,lvar,uvar,name,Model=LSQRModel,\
                             txt='False', npz='True')
    print lsqpr.name
    J = sp(matrix=as_llmat(Q))
    e1 = np.ones(J.shape[0])
    e2 = np.ones(J.shape[1])
    print 'J.shape = ', J.getShape()

    print 'Testing PysparseLinearOperator:'
    op = PysparseLinearOperator(J)
    print 'op.shape = ', op.shape
    print 'op.T.shape = ', op.T.shape
    print 'op * e2 = ', op * e2
    print "op.T * e1 = ", op.T * e1
    print 'op.T.T * e2 = ', op.T.T * e2
    print 'op.T.T.T * e1 = ', op.T.T.T * e1
    print 'With call:'
    print 'op(e2) = ', op(e2)
    print 'op.T(e1) = ', op.T(e1)
    print 'op.T.T is op : ', (op.T.T is op)
    print
    print 'Testing LinearOperator:'
    op = LinearOperator(J.shape[1], J.shape[0],
                        lambda v: J*v,
                        matvec_transp=lambda u: u*J)
    print 'op.shape = ', op.shape
    print 'op.T.shape = ', op.T.shape
    print 'op * e2 = ', op * e2
    print 'e1.shape = ', e1.shape
    print 'op.T * e1 = ', op.T * e1
    print 'op.T.T * e2 = ', op.T.T * e2
    print 'op(e2) = ', op(e2)
    print 'op.T(e1) = ', op.T(e1)
    print 'op.T.T is op : ', (op.T.T is op)
    print
    op2 = op.T * op
    print 'op2 * e2 = ', op2 * e2
    print 'op.T * (op * e2) = ', op.T * (op * e2)
    print 'op2 is symmetric: ', check_symmetric(op2)
    op3 = op * op.T
    print 'op3 * e1 = ', op3 * e1
    print 'op * (op.T * e1) = ', op * (op.T * e1)
    print 'op3 is symmetric: ', check_symmetric(op3)
    print
    print 'Testing negative operator:'
    nop = -op
    print op * e2
    print nop * e2
    I = LinearOperator(nargin=4, nargout=4,
                       matvec=lambda v: v, symmetric=True)
    
def exampleliop(n,m):
    #    Q,B,d,c,lcon,ucon,lvar,uvar,name = first_class_tp(3,2,2)#(2,2,3 )
    Q,B,d,c,lcon,ucon,lvar,uvar,name = fifth_class_tp(n,m)
    
    Q = sp(matrix=as_llmat(Q))
    B = sp(matrix=as_llmat(B))
    #print np.identity(Q.shape[0])[0,:]*Q
    #print Q[0,:]
  
    Q = PysparseLinearOperator(Q)
    B = PysparseLinearOperator(B)
    #print as_llmat(FormEntireMatrix(Q.shape[1],Q.shape[0],Q))
    #C = LinearOperator(nargin=4, nargout=4,
                       #matvec=lambda v: v, symmetric=True)

    lsqpr = LSQRModel(Q=Q, B=B, d=d, c=c, Lcon=lcon, Ucon=ucon, Lvar=lvar,\
                    Uvar=uvar, name='test')
    #print Q.shape,B.shape,d.shape,c.shape,lcon.shape,ucon.shape,lvar.shape

    #lsqpr = LSQRModel(Q=lsqp.Q, B=lsqp.B, d=lsqp.d, c=lsqp.c, Lcon=lsqp.Lcon,\
                      #Ucon=lsqp.Ucon, Lvar=lsqp.Lvar,Uvar=lsqp.Uvar, name=lsqp.name)
    return lsqpr
def print_matrix(operator_object):
    m,n = operator_object.shape
    print as_llmat(operator_object.to_array())
    return
def l1_ls_class_tp(p=0,n=0,delta = 1.0e-05):
    Q,y = l1_ls_itre(p,n)
    y = sprandvec(p,30)
    d = Q*y
    d = np.array(d)[:,0]
    print "Number of non zero in original",sum(y)

    
    c = np.zeros(n)
    c = np.concatenate((np.zeros(n),np.ones(n)*delta), axis=1)
    ucon = np.zeros(2*n)
    lcon = -np.ones(2*n)*inf
    uvar = np.ones(2*n)*inf
    lvar = -np.ones(2*n)*inf

    I = IdentityOperator(n, symmetric=True)
    # Build [ I  -I]
    #       [-I  -I]
    B = BlockLinearOperator([[I, -I], [-I]], symmetric=True)

    Q_ = ZeroOperator(n,p)
    new_Q = BlockLinearOperator([[Q,Q_]])
    p, n = new_Q.shape
    m, n = B.shape
    name = str(p)+'_'+str(n)+'_'+str(m)+'_l1_ls'
    lsqpr = LSQRModel(Q=new_Q, B=B, d=d, c=c, Lcon=lcon, Ucon=ucon, Lvar=lvar,\
                    Uvar=uvar, name=name)
    return lsqpr

def nnz_elements(regqp,tol=1e-4):
##    n = x0.shape[0]
    print "non zero % with Tol  :",tol
##    xmax0 = tol*numpy.linalg.norm(x0,numpy.inf)
##    nnz0 = len(x0[abs(x0)>=xmax0])*1.0/len(x0)*100.
    x = regqp
    n = x.shape[0]
    xmax = tol*numpy.linalg.norm(x,numpy.inf)  
    nnz = len(x[abs(x)>= xmax])*1.0/len(x)*100
    print "Non zero elements are: ",nnz,"%"
    return nnz

def sprandvec(m,n):
    """
    Must be m>=n
    """
    i, d = divmod(m*(n/100.), 1)
    print n,"% of the original vector with dimention ",m,"is",i
    n_ = int(i)
    v = np.zeros([1,m])
    r = np.random.permutation(range(m))
    r = r[:n_]
    for i in range(n_):
	v[0,r[i]] = 1 
    _folder = str(os.getcwd()) + '/binary/'
    if not os.path.isdir(_folder):
                os.makedirs(_folder)
    np.savez(_folder+str(n)+'_'+str(m), v.T)
    nnz_elements(v.T,tol=1e-4)
    return v.T
if __name__ == "__main__":
    from pykrylov.linop import LinearOperator
    import logging
    import sys
    from numpy.linalg import norm
    ls = l1_ls_class_tp(3,6)
    remove_type_file()