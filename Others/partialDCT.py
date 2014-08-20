from numpy import logical_xor
import numpy as np
import copy
from dctt import dctt,idctt

class partialDCT():
    """
    """
    def __init__(self, n=0, m=0, J=[]):
        
        self.adjoint =0
        self.n = n
        self.m = m
        self.J = J[0:m]
##        self.__T = self.T()
        
    def __mul__(self,other):
        ""
        if  self.adjoint == 0: #A*x
            #res = dctt(other)
            return dctt(other)[self.J]
        else: #At*x
            z = np.zeros([self.n,1])
            z[self.J] = other[self.J]
            return idctt(z)
    
    def T(self):
        At = copy.copy(self)
        At.adjoint = logical_xor(At.adjoint,1)
        return At
        
    def sign(self, x):
        return 1 if x >= 0 else -1

def test (m=128,n=1024):
    "n is signal dimension and m is number of measurements"

    J = range(n)
    J = J[0:m]     # or we can choose m randomly indices : 
                   # J = np.random.permutation(range(n)) 
    print J
    A = partialDCT(n,m,J)
    # spiky signal generation
    T = min(m,n)-1 # number of spikes
    x0 = np.zeros([n,1]);
    q = np.random.permutation(range(n)) #or s=list(range(5)) random.shuffle(s)
    x0[q[0:T]]= np.sign(np.random.rand(T,1))
    #x0[J[0:T]] = np.sign(np.reshape(np.arange(1,T+1),(T,1)))
    # noisy observations
    sigma = 0.01  # noise standard deviation
    y = A*x0 + sigma*np.reshape(np.arange(1,m+1),(m,1))
    Lambda = 0.01  # regularization parameter
    rel_tol = 0.01  # relative target duality gap
    print x0
    print A*x0
    print A.T()*x0 # transpose of A
    #print x0,idctt(x0)
##    print A*x0
    return A,y
if __name__ == "__main__":
    A,y=test(3,6)
##    print A
##    print y