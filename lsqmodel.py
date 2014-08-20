# lsqmodel.py
# Define an abstract class to represent a general
# least-squares problem.
#
# M.Dehghani, Montreal  2012.

from pykrylov.linop import *
from nlpy.krylov import SimpleLinearOperator, ReducedLinearOperator, SymmetricallyReducedLinearOperator
from nlpy.model.nlp import NLPModel
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
from pysparse.sparse import PysparseIdentityMatrix as eye

import numpy as np

__docformat__ = 'restructuredtext'


class LSQModel(NLPModel):
    """
    LSQModel creates an instance of least-squares problem.
           minimize    c'x + 1/2|Qx-d|^2

           subject to  LCon <= Bx <= UCon,                       (LSQP)
                       LVar <=  x <= UVar,

    Where `Q`  is an (p x n) and `B`  is an (m x n) matrices.
    """

    def __init__(self, Q, B, d, c, Lcon, Ucon, Lvar, Uvar, **kwargs):

        NLPModel.__init__(self, n=Q.shape[1], m=B.shape[0], Lcon=Lcon, \
                          Ucon=Ucon, Lvar=Lvar, Uvar=Uvar, **kwargs)

        #Initialize the parameters of least squares
        self.Q = Q
        self.B = B
        "Return a contiguous array in memory (C order)."
        self.d = np.ascontiguousarray(d, dtype=float)
        self.c = np.ascontiguousarray(c, dtype=float)

    def close(self):
        """
        Not needed anymore. Here for backward compatibility with
        original LSQModel interface.
        """
        pass

    def obj(self, x):
        """
        Evaluate objective function value at x for the least squares
        problem and returns a floating-point number.
        """
        x = np.ascontiguousarray(x, dtype=float)
        Qx = self.Q * x
        lsq_term = Qx - self.d
        f = np.dot(self.c, x) + 0.5 * np.dot(lsq_term, lsq_term)
        return f

    def A(self):
        " To obtain constraint matrix of the least squares problem."
        Bmat = self.B
        return  Bmat

    def jac(self, x=None):
        """
        Evaluate Jacobian of constraints.
        In the linear case of constraints this is
        equal to coefficient matrix of the constraints.
        """
        J = self.A()
        return J

    def hess(self, x=None, z=None, obj_num=0):
        "Return the Hessian of the problem which is equal to Q'Q."
        H = self.Q.T*self.Q
        return H

    def cost(self):
        "Return a Numpy array as cost vector."
        return  self.c

    def cons(self, x):
        "Evaluate vector of constraints at x and returns a Numpy array."
        x = np.ascontiguousarray(x, dtype=float)
        Z = ZeroOperator()
        B = Z.ReducedLinearOperator(self.B, x.size, x.size)
        Bx = B * x
        return Bx

    def grad(self, x):
        "Evaluate objective gradient at x and returns a Numpy array."
        x = np.ascontiguousarray(x, dtype=float)
        Q = self.Q
        Qx = Q * x
        Qxd = Qx - self.d
        Qxd = Q.T * Qxd
        cx = self.c + Qxd
        return cx


class LSQRModel(LSQModel):
    """
    LSQRModel creates an instance of least-squares problem
           minimize    c'x + 1/2 |r|^2
           subject to  Qx + r = d,
                       LCon <= Bx <= UCon,                       (LSQR)
                       LVar <=  x <= UVar,

    Where `Q`  is a (p x n) matrix and `B`  is an (m x n) matrix.
    The variables are assumed to be ordered as (x,r).
    """

    def __init__(self, Q, B, d, c, Lcon, Ucon, Lvar, Uvar, **kwargs):

        p, n = Q.shape
        m = B.shape[0]
        _Lcon = np.concatenate((d, Lcon))
        _Ucon = np.concatenate((d, Ucon))
        _Lvar = np.empty(n+p)
        _Lvar[:n] = Lvar
        _Lvar[n:] = -np.inf
        _Uvar = np.empty(n+p)
        _Uvar[:n] = Uvar
        _Uvar[n:] = np.inf

        NLPModel.__init__(self, n=n+p, m=m+p, Lcon=_Lcon, \
                          Ucon=_Ucon, Lvar=_Lvar, Uvar=_Uvar, **kwargs)

        # Initialize the parameters of least squares
        self.Q = Q
        self.B = B
        self.nx = n
        self.nr = p
        self.mx = m
        "Return a contiguous array in memory (C order)."
        self.d = np.ascontiguousarray(d, dtype=float)
        self.c = np.ascontiguousarray(c, dtype=float)

    def obj(self, xr):
        nx, nr = self.nx, self.nr
        x = xr[:nx]
        r = xr[nx:]
        return np.dot(self.c, x) + 0.5 * np.dot(r, r)

    def grad(self, xr):
        nx = self.nx
        r = xr[nx:]
        return np.concatenate((self.c, r))

    #def hess(self, xr= None, *args, **kwargs):
        #nx, nr = self.nx, self.nr
        #H =  PysparseLinearOperator(eye(self.nvar))
        #return H

    def cons(self, xr):
        nx, nr = self.nx, self.nr
        x = xr[:nx]
        r = xr[nx:]
        c = np.empty(self.m)
        c[:nr] = self.Q * x + r
        c[nr:] = self.B * x
        return c

    def jprod(self,x,u):
        """
          nx  nr
         [Q   I][u]  [Qu + v]   nr=4
                   = 
         [B   0][v]  [Bu    ]   mx=3
        """
        p = np.zeros(self.nr + self.mx)
        p[:self.nr] = self.Q*u[:self.nx] + u[self.nx:]
        p[self.nr:] = self.B*u[:self.nx]
        return p

    def jtprod(self,x,u):
        """
         [Q^T   B^T][u]  [Q^Tu + B^Tv]
                       = 
         [I       0][v]  [u          ]
        """
        
        p = np.zeros(self.nr + self.nx)
        p[:self.nx]= self.Q.T*u[:self.nr] + self.B.T*u[self.nr:]
        p[self.nx:]= u[:self.nr]
        return p
    
    def jac(self, x, **kwargs):
        return SimpleLinearOperator(self.n, self.m, symmetric=False,
                         matvec=lambda u: self.jprod(x,u,**kwargs),
                         matvec_transp=lambda u: self.jtprod(x,u,**kwargs))
    # Evaluate matrix-vector product between
    # the Hessian of the Lagrangian and a vector
    def hprod(self, x, y, v, **kwargs):
        return v
    
    def hess(self, x, z, **kwargs):
        return SimpleLinearOperator(self.nx+self.nr, self.nx+self.nr, symmetric=True,
                         matvec=lambda u: self.hprod(x,z,u,**kwargs))
