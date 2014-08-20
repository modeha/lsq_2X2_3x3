import numpy as np
from numpy import *
from math import sqrt, cos, pi
from cmath import exp
from scipy.sparse import spdiags
from scipy.fftpack import fft,ifft

class data(object):
    """
    """
    def __init__(self, name=None, signal=None,\
                 opDict=0, B=None, b=None, x=None):
        self.name = name
        self.signal = signal
        self.opDict = opDict
        self.B = B
        self.b = b
        self.x = x
        
    def checkDimensions(self,m,n,x,mode):
        ""
        mx,nx = x.shape
        if mode == 0:
            return
        if (mx== 1) and (nx == 1):
            if (m != 1) or (n != 1):
                error('Operator-scalar multiplication not yet supported')
        if mode == 1:
            if (nx != n):
                error('Incompatible dimensions')
        if (mx != 1):
            error('Operator-matrix multiplication not yet supported')
        else:
            if (nx != m):
                error('Incompatible dimensions')
            if (mx != 1):
                error('Operator-matrix multiplication not yet supported')
        return
    
    def opFFT(self,n):
        
        """
        OPFFT  One-dimensional fast Fourier transform (FFT).
        OPFFT(N) create a one-dimensional normalized Fourier transform
        operator for vectors of length N.
        """
     #   return lambda x: x + n
        op = lambda x,mode : self.opFFT_intrnl(n,x,mode)
        return op
    
    def opFFT_intrnl(self,n,x,mode):
        
        self.checkDimensions(n,n,x,mode)
        mx,nx = x.shape
        lenghth_x = max(nx,mx)
        if mode == 0:
            y = [n,n,[1,1,1,1],['FFT']]
        elif mode == 1:
            y = fft(x) / sqrt(length_x)
        else:
            y = ifft(x) * sqrt(length_x)
        return y
    
    def opHeaviside(self,n,s=0):
        """
            OPHEAVISIDE  Heaviside operator
            OPHEAVISIDE(N,S) creates an operator for multiplication by an
            N by N Heaviside matrix. These matrices have ones below and on
            the diagonal and zeros elsewhere. S is a flag indicating
            whether the columns should be scaled to unit Euclidean
            norm. By default the columns are unnormalized.
        """
        op = lambda x,mode : self.opHeaviside_intrnl(n,s,x,mode)
        return op
    
    def opHeaviside_intrnl(self,n,s,x,mode):
        ""
        self.checkDimensions(n,n,x,mode)
        if mode == 0:
            y = [n,n,[0,1,0,1],['Heaviside',n]]
        elif mode == 1:
            # Scale if normalized columns requested
            if s != 0:
                x = 1./sqrt(arange(n,0,-1))*x
            y = cumsum(x);
        else:
            y = cumsum(x)
            ym = y[-1]
            y[1:-1] = ym - y[0:-2]
            y[0] = ym
            # Scale if normalized columns requested
            if s != 0:
                y = 1./sqrt(range(n,0,-1))*y
        return y
    
if __name__ == "__main__":
    ob = data()
    print ob.opHeaviside(4,1)