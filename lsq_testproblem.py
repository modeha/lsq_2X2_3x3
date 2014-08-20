
import numpy  as np
import os

def remove_type_file(type='.pyc',path= (os.getcwd())):
    
    for file in os.listdir(path):
        if str(file[-len(type):]) ==type:
            os.remove(path+'/'+file)

def nnz_elements(regqp,tol=1e-4):
##    n = x0.shape[0]
    print "non zero % with Tol  :",tol
##    xmax0 = tol*np.linalg.norm(x0,np.inf)
##    nnz0 = len(x0[abs(x0)>=xmax0])*1.0/len(x0)*100.
    x = regqp
    n = x.shape[0]
    xmax = max(tol,tol*np.linalg.norm(x,np.inf))  
    nnz = len(x[abs(x)>= xmax])*1.0/len(x)*100
    print "Non zero elements are: ",nnz,"%"
    # np.set_printoptions(threshold='nan')
    # print x
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
    # _folder = str(os.getcwd()) + '/binary/'
    # if not os.path.isdir(_folder):
    #             os.makedirs(_folder)
    # np.savez(_folder+str(n)+'_'+str(m), v.T)
    nnz_elements(v.T,tol=1e-4)
    return v.T

if __name__ == "__main__":
    from pykrylov.linop import LinearOperator
    import logging
    import sys
    import numpy as np
    from numpy.linalg import norm
    remove_type_file()