import numpy as np
from numpy import *
from math import sqrt, cos, pi
from cmath import exp
from scipy.sparse import spdiags
from pykrylov.linop import LinearOperator
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
from toolslsq import as_llmat
from pykrylov.linop import *
from lsqmodel import LSQModel, LSQRModel
from lsq_testproblem import *

#from toolslsq import as_llmat


def arrayexp(n):
    """Returns the elementwise antilog of the real array x.  We try to
    exponentiate with numpy.exp() and, if that fails, with python's
    math.exp().  numpy.exp() is about 10 times faster but throws an
    OverflowError exception for numerical underflow (e.g. exp(-800),
    whereas python's math.exp() just returns zero, which is much more
    helpful.
    """
    x = range(n)
    ww = np.empty(len(x), complex)
    for j in range(len(x)):
        ww[j] = (exp(-1j*x[j]*pi/(2*n))/sqrt(2*n))
    ww[0] = ww[0]/sqrt(2)
    return np.array([ww],dtype=complex)

def iarrayexp(n):
    """Returns the elementwise antilog of the real array x.  We try to
    exponentiate with numpy.exp() and, if that fails, with python's
    math.exp().  numpy.exp() is about 10 times faster but throws an
    OverflowError exception for numerical underflow (e.g. exp(-800),
    whereas python's math.exp() just returns zero, which is much more
    helpful.
    """
    x = range(n)
    ww = np.empty(len(x), complex)
    for j in range(len(x)):
        ww[j] = sqrt(2*n)*exp(1j*x[j]*pi/(2*n))
    return np.array([ww],dtype=complex)


def dctt1(a):
    """ dct  Discrete cosine transform.
    y = dct(a) returns the discrete cosine transform of a.
    The vector y is the same size as `a` and contains the
    discrete cosine transform coefficients.
    """
    if len(a.shape)==1:
        a = a.reshape(a.size,1)
    n,m = a.shape
    aa = a[:,:]
    #Compute weights to multiply DFT coefficients
    ww = arrayexp(n)
    if n%2 == 1:
        y = np.zeros([2*n,m])
        y[:n,:] = aa
        y[n:2*n,:] = np.flipud(aa)
        # Compute the FFT and keep the appropriate portion:
        yy = np.fft.fft(y,axis=0)
        yy = yy[:n,:]
    else:
        # Re-order the elements of the columns of x
        y = np.concatenate((aa[np.ix_(range(0,n,2))],\
                            aa[np.ix_(range(1,n,2)[::-1])]), axis=0)
        yy = np.fft.fft(y,axis=0)
        ww = 2*ww  # Double the weights for even-length case 

    wy = np.empty([n,m], complex)
    for j in range(m):
        wy[:,j]  = ww
    # Multiply FFT by weights:
    b = np.multiply(wy,yy)
    
    return b[:n,:m].real

def idctt1(b):
    """ idct  inverts the Discrete cosine transform.
    y = dct(a) returns the inverse discrete cosine transform of a.
    The vector y is the same size as a and contains the
    discrete cosine transform coefficients.
    Also returning the original vector if y was obtained using y = DCT(a).
 
    """
    if len(b.shape)==1:
        b = b.reshape(b.shape[0],1)
    n,m = b.shape
    bb = b[:,:]
    #Compute weights to multiply DFT coefficients
    ww = iarrayexp(n)
    if n%2 == 1:
        ww[0][0] = ww[0][0]*sqrt(2)
        wy = np.empty([n,m], complex)
        for j in range(m):
            wy[:,j]  = ww
        W = wy
        yy = np.zeros([2*n,m],complex)
        yy[:n,:] = np.multiply(W,bb)
        yy[n+1:2*n,:] = np.multiply(complex(0,-1)*W[1:n,:],np.flipud(bb[1:,:]))
        # Compute the FFT and keep the appropriate portion:
        y = np.fft.ifft(yy,axis=0)
        a = y[:n,:]
    else:
        ww[0][0] = ww[0][0]/sqrt(2)
        wy = np.empty([n,m], complex)
        for j in range(m):
            wy[:,j]  = ww
        W = wy
        yy = np.multiply(wy,bb)
        y = np.fft.ifft(yy,axis=0)
       
        a = np.zeros([n,m],complex)
        a[np.ix_(range(0,n,2)),:] = y[0:n/2,:]
        a[np.ix_(range(1,n,2)),:] = y[range(n-1,(n-1)/2,-1),:]
        # Re-order the elements of the columns of x
    return a[:n,:m].real

def dctt(a):
    """ dctt 2-D discrete cosine transform.
    B = dctt(A) returns the discrete cosine transform of A.
    The matrix B is the same size as A and contains the
    discrete cosine transform coefficients.
    """
    if len(a.shape)==1:
        return (dctt1(a))[0,:]
    return dctt1(a)

def idctt(a):
    """ dctt 2-D discrete cosine transform.
    B = dctt(A) returns the discrete cosine transform of A.
    The matrix B is the same size as A and contains the
    discrete cosine transform coefficients.
    """
    if len(a.shape)==1:
        return (idctt1(a))[0,:]
    return idctt1(a)
##
##    #idctt1(idctt1(a).T).T
##    return idctt1(idctt1(a).T)[:,0]

def sign(x): 
    return 1 if x >= 0 else -1


def z_v(n,J,v):
    z = np.zeros([n,1])
    if  len(v.shape)==1:
        v = v.reshape(v.shape[0],1)
    z[J] = v[J]
    return z[:,0]

def partial_DCT(n = 10, m = 4, delta = 1.0e-05):
    "n is signal dimension and m is number of measurements"
    #n  signal dimension
    #m  number of measurements
    z = np.zeros([n,1])
    J = np.random.permutation(range(n)) # m randomly chosen indices
    J = np.array(range(n))
    
    # generate the m*n partial DCT matrix whose m rows are
    # the rows of the n*n DCT matrix at the indices specified by J

    A = LinearOperator(nargin=m, nargout=n, matvec=lambda v: dctt(v),
                       matvec_transp=lambda v: idctt(z_v(n,J,v)))

    # spiky signal generation
    T = min(m,n)-1 # number of spikes
    x0 = np.zeros([n,1]);
    q = np.random.permutation(range(n)) #or s=list(range(5)) random.shuffle(s)
    x0[q[0:T]]= np.sign(np.random.rand(T,1))
    #x0[J[0:T]] = np.sign(np.reshape(np.arange(1,T+1),(T,1)))
    # noisy observations
    sigma = 0.01  # noise standard deviation
    y = x0  #+ sigma*np.reshape(np.arange(1,m+1),(m,1))

    Q = A
    #d = y[:,0]; 
    p = n; n = m

    #Q = np.tril(np.ones((p, n), dtype=int), 0)+p*n*np.eye(p,n)
    y = sprandvec(p,30)
    d = Q*y
    d = np.array(d)[:,0]
    print "Number of non zero in original",sum(y)
    #numpy.set_printoptions(threshold='nan')
    #print y
    
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

if __name__ == "__main__":
    from toolslsq import *
    from pykrylov.linop import LinearOperator
    from pykrylov.linop import *
    X = 10 * np.random.rand(2600,3400)
    X = np.array([[0.2867,0.8133,0.2936],[0.9721,0.7958,0.3033],
                  [0.6198,0.6210,0.1496],[0.2760,0.8500,0.6658]])
##    X = np.array([[0.2867,0.8133,0.2936]])
##    X = np.array([[1.,2.,3.,4.]])

    m=3;n=4
    A = LinearOperator(nargin=m, nargout=n, matvec=lambda v: dctt1(v),
                       matvec_transp=lambda v: idctt(z_v(n,J,v)))
    A = LinearOperator(nargin=m, nargout=n, matvec=lambda v: dctt(v),
                       matvec_transp=lambda v: idctt(z_v(n,J,v)))

    #print A

    print idctt(X)
    print dctt(X)
##    
##    >> dct(A)
##
##ans =
##
##    1.0773    1.5400    0.7061
##    0.1023    0.0233   -0.2016
##   -0.5146    0.1233    0.2532
##   -0.2273   -0.1241   -0.2011
##
##>> idct(A)
##
##ans =
##
##    1.1630    1.4670    0.5999
##   -0.0838   -0.2438   -0.2809
##   -0.2493    0.4361    0.4249
##   -0.2565   -0.0327   -0.1567
##
##>> A
##
##A =
##
##    0.2867    0.8133    0.2936
##    0.9721    0.7958    0.3033
##    0.6198    0.6210    0.1496
##    0.2760    0.8500    0.6658
##
##>> 
      
##    A,y = l1_ls_itre(m = 3, n = 4)
##    a=A*np.ones([A.shape[1],1])
##    print A.to_array()
