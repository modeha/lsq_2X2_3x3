
from lsqmodel import LSQModel

from pysparse.sparse import spmatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix
from pysparse.sparse import pysparseMatrix as sp

import numpy as np


def lsq(lsq_ff):
    """ 

    :param lsq_ff:
    Convert the LSQP in the First Form(FF) ::
    
           minimize    c'x + 1/2|Qx-d|^2
           subject to  L <= Bx <= U,                       (LSQP-FF)
                       l <=  x <= u,
    to the Second Form (SF):: 
    
            minimize    c'x +1/2|r|^2
            subject to. [d] <= [Q  I][r] <= [d],
                        [L] <= [B  0][x] <= [U],            (LSQP-SF)
                        [l] <=       [x] <= [u],
                     -[inf] <=       [r] <= [inf].
    """
     
     
    p,n = lsq_ff.Q.shape
    m,n = lsq_ff.B.shape
     
    new_B = spmatrix.ll_mat(m+p, n+p, m+n+2*p+lsq_ff.B.nnz+lsq_ff.Q.nnz)
    new_B[:p,:n] = lsq_ff.Q
    new_B[p:,:n] = lsq_ff.B
    
    new_B.put(1, range(p), range(n,n+p))
   
    new_Lcon = np.zeros(p+m)    
    new_Lcon[:p] = lsq_ff.d   
    new_Lcon[p:] = lsq_ff.Lcon
    
    new_Ucon = np.zeros(p+m)    
    new_Ucon[:p] =  lsq_ff.d
    new_Ucon[p:] = lsq_ff.Ucon
    
    new_Lvar = -np.inf * np.ones(n+p)    
    new_Lvar[:n] = lsq_ff.Lvar
    
    new_Uvar = np.inf * np.ones(n+p)
    new_Uvar[:n] = lsq_ff.Uvar   
    
    new_Q = PysparseMatrix(nrow=n+p, ncol=n+p,\
                           sizeHint=p)
    new_Q.put(1, range(n,n+p), range(n,n+p))

    new_d = np.zeros(n+p)
    
    new_c = np.zeros(n+p)
    new_c[:n] = lsq_ff.c
    
    return LSQModel(Q=new_Q, B=new_B, d=new_d, c= new_c, Lcon=new_Lcon, \
                    Ucon=new_Ucon, Lvar=new_Lvar, Uvar=new_Uvar,
                    name= lsq_ff.name, dimQB=(p,n,m))

if __name__ == '__main__':
    from lsq_testproblem import  *
    from slack_nlp import SlackFrameworkNLP as SlackFramework
    import numpy as np
    Q,B,d,c,lcon,ucon,lvar,uvar,name = first_class_tp(2,1,3)
    
    #print "***************B**************"
    lsqp = lsq_testproblem(Q,B,d,c,lcon,ucon,lvar,uvar,name )
    #print lsq_ffr.B
    #print "*************Q****************"
    #print lsq_ffr.Q
    #lsq_ffr2= lsq(lsq_ffr)
    #print "***************B**************"
    #print lsq_ffr2.B
    #print "***************Q**************"
    #print lsq_ffr2.Q
    lsqr = lsq(lsqp)
    #lsqr.display_basic_info()
    lsqpsf = SlackFramework(lsqr)
    A = lsqpsf.jac(lsqpsf.x0)
    #print lsq_ffr2.dim_QB()
    #lsqpsf = SlackFramework(lsq_ffr2)
    #print lsqpsf.original_m,lsqpsf.original_m
    #print lsqpsf.m,lsqpsf.n
    print isinstance(A, PysparseMatrix)
    A.exportMmf('pysA')
    #import scipy.io as sc
    #sc.mmwrite('/tmp/myarray',A)
    #sc.mmread('/tmp/myarray')
