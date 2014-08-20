from lsqmodel import LSQModel, LSQRModel
from Others.toolslsq import *
from Others.dctt import dctt, test, idctt,l1_ls_itre

from numpy import fft, array, arange, zeros, dot, transpose
from pysparse import spmatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
from pysparse.sparse import PysparseIdentityMatrix as eye

from Others.toolslsq import as_llmat, contiguous_array
from pykrylov.tools import check_symmetric
from pykrylov.linop import *
import sys

from math import sqrt, cos, pi
import Others.copy_past as cp
import numpy as np
from scipy import Infinity as inf
import shutil
from shutil import copytree, ignore_patterns
import os,numpy,glob
import random


_folder = str(os.getcwd()) + '/output/tp_npz/'


def  dcttm(B):
    """


    :rtype : object
    :param B:
    :return:
    """
    B = as_llmat(B)
    B = PysparseMatrix(matrix=B)
    [M,N] = B.shape
    A = np.zeros([M,N])
    for m in xrange(0, M):
        for n in xrange(0, N):
            c = alpha(m,M)*alpha(n,N)
            a = g(M,m)
            b = g(N,n)
            
            Bb = B*b.T[:,0]
            A[m][n] = c*np.dot(a,Bb)#dot(a,B.T)#*b.T
    return A

def  alpha(p,n):
    """

    :param p:
    :param n:
    :return:
    """
    if p==0:
        m = sqrt(1/float(n))
    else:
        m = sqrt(2/float(n))
    return m

def  g(M,p):
    """

    :param M:
    :param p:
    :return:
    """
    b = np.zeros([1,M])
    for x in xrange(0, M):
        b[0][x] = float(cos(p*pi * (2 * x  + 1) / (2 * M)))
    return b


def save_to_npz(Q, B, d, c, Lcon, Ucon, Lvar, Uvar, name='Generic'):
    "save least squares datas to the .npz file"
    arrs = {'Q': Q, 
            'B': B,
            'd': d,
            'c': c,
            'Lcon': Lcon,
            'Ucon': Ucon,
            'Lvar': Lvar,
            'Uvar': Uvar,
            'name': str(name)
            }
    if not os.path.isdir(_folder):
        os.makedirs(_folder)
    np.savez(_folder+name[:-4], **arrs)
    return

def load_npz_to_dic(file_name):
    " read a .npz file return a dictionary"
    npz_file = np.load(file_name+'.npz', 'rb')
    return npz_file

def npz_to_lsqobj(filename, Model=LSQModel):
    "read a .npz file and return an lsq object"
    datas = load_npz_to_dic(filename)
    lsq_obj = Model(Q=datas['Q'], B=datas['B'], d=datas['d'], c=datas['c'],\
                       Lcon=datas['Lcon'], Ucon=datas['Ucon'],\
                       Lvar=datas['Lvar'], Uvar=datas['Uvar'],\
                       name=str(datas['name']))
    return lsq_obj

def lsq_tp_generator(Q, B, d, c, lcon, ucon, lvar, uvar, name,\
                     Model=LSQModel, txt='False', npz='False'):
    "Genarate npz and txt files with given datas and return a lsq object."
    if txt =='True':
	 # Save generated test problem to compare with a solver
	create_ample_file(Q, B, d, c, lcon, ucon, lvar, uvar,name)
    if npz is 'True':
	save_to_npz(Q=Q, B=B, d=d, c=c, Lcon=lcon, Ucon=ucon, Lvar=lvar,\
	            Uvar=uvar, name=name)
    
    lsq_obj = Model(Q=Q, B=B, d=d, c=c, Lcon=lcon, Ucon=ucon, Lvar=lvar,\
                       Uvar=uvar, name=name)
    
    return  lsq_obj

def ampl_matrix(A):
    " Transfers matrix to AMPL format."
    
    m, n = A.shape 
    row = np.arange(1, n+1)
    B = np.row_stack([row,A])

    p, q = B.shape
    
    if n==1:
        return np.column_stack([range(1,m+1),A])
    else :
        col = range(p)
        return np.column_stack([col,B])

def add_hedder(filename=None, val=None, var=None, rowcol=None):
    "Adds the headers in AMPL data file"
    format = 'a+'
    line1 = None
    line2 = None
    
    if rowcol:
        line1 = "set " + rowcol + " := " + "1.." + var + " ;\n"
    if var:
        line2 = "param " + var + " := " + str(val) + " ;\n"
    
    fwrite = open(filename, format)
    if line2: print >>fwrite,line2
    if line1: print >>fwrite,line1 
    return

def write_ampl(filename, matrix_name, matrix):
    "Creates AMPL data file."
    format = 'a+'
    path = str(os.getcwd()) + '/tmp.txt'
    np.savetxt( path, matrix, fmt="%8.8G")
    fread =  open(path,'r')
    fwrite = open(filename, format)
    
    if matrix_name in ['b','c','d','ucon','lcon','uvar','lvar']:
        print >>fwrite, "param " + str(matrix_name) + ":="
    else:
        print >>fwrite, "param " + str(matrix_name) + ":"
    i=1
    for line in fread:
        line = line.replace('INF', 'Infinity ')
        fwrite = open(filename, format)
        if i==1 and len(line)<20:
            i+=1
            line = line.replace(' 0 ', ' ')
            print >> fwrite, line
            
        elif i==1 and len(line)>20:
            i+=1
            line = line.replace(' 0 ', ' ')
            print >> fwrite, line.replace("\n", " :=\n")
        else:
            print >> fwrite, line
    fwrite = open(filename, format)
    print >> fwrite, ";\n"
    if os.path.exists(path):
        os.remove(path)
    return

def create_ample_file(Q=None, B=None, d=None, c=None,\
                      lcon=None, ucon=None, lvar=None, uvar=None,name=None):
    "Genarate an AMPL file with least squares datas"
    tmp = 'ampl.txt'
    if name is None:
        filename = 'ampl_test.txt'
    else:
        filename = name

    main_path = str(os.getcwd())
    folder = main_path + '/output/tp_txt/'
    if not os.path.isdir(folder):
        os.makedirs(folder)
    tmp_folder = main_path + '/tmp/'
    if not os.path.isdir(tmp_folder):
        os.makedirs(tmp_folder)
    
    path_ = folder + filename
    path_tmp = tmp_folder + tmp

    if os.path.exists(path_):
        os.remove(path_)
    if os.path.exists(path_tmp):
        os.remove(path_tmp)
    m, n = B.shape
    p, n = Q.shape
    
    add_hedder(path_, n, 'n', 'column_A_Q')
    add_hedder(path_, m, 'm', 'row_A')
    add_hedder(path_, p, 'p', 'row_Q')
    
    write_ampl(path_, 'Q', ampl_matrix(Q))
    write_ampl(path_, 'B', ampl_matrix(B))
    write_ampl(path_, 'd', ampl_matrix(change_to_vec(d)))
    write_ampl(path_, 'c', ampl_matrix(change_to_vec(c)))
    
    write_ampl(path_, 'lvar', ampl_matrix(change_to_vec(lvar)))
    write_ampl(path_, 'uvar', ampl_matrix(change_to_vec(uvar)))
    write_ampl(path_, 'lcon', ampl_matrix(change_to_vec(lcon)))
    write_ampl(path_, 'ucon', ampl_matrix(change_to_vec(ucon)))
    
    cp.paste_file(cp.copy_file(path_), tmp_folder)
    os.rename(main_path +'/tmp/'+name, main_path +'/tmp/'+'ampl.txt')
    
    if os.path.exists(tmp_folder):
        
        shutil.rmtree(tmp_folder)
    return

def change_to_vec(c):
        dim = c.shape
        m =1
        n = dim[0]
        vec = np.zeros([n,1])
        for i in range(n):
                vec[i] = c[i]
        return vec
    
def remove_type_file(type='.pyc',path= (os.getcwd())):
    
    for file in os.listdir(path):
        if str(file[-len(type):]) ==type:
            os.remove(path+'/'+file)

def first_class_tp(nvar=18, prowQ=9, mcon=4 ):

    eps = sys.float_info.epsilon
    inf = 1/eps
    """First class of test problems we choose random data 
    generated data such that x=(1,...,1) be the exact solution."""
    n = nvar + prowQ + mcon
    p = prowQ + mcon
    m = mcon
    c = np.zeros(n)
    d = np.zeros(p)
    foo = [inf, eps, 1]
    uvar =[]; lvar = []
    for i in range(n):
	uvar.append(random.choice(foo))
	lvar.append(-random.choice(foo))
    ucon = np.ones(m)*inf
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
    lcon = -b*10000
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
    eps = sys.float_info.epsilon
    #inf = 1/eps    
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
    
    #foo = [eps,1/eps, 1]
    

    #uvar =[]; lvar = []
    #for i in range(2*n):
	    #uvar.append(random.choice(foo))
	    #lvar.append(-random.choice(foo))    

    #print 'uvar',uvar
    #print 'lvar',lvar
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

def mat_py(path= (os.getcwd())):
    "convert matlab to python."
    Q,B,d,c,Lcon,Ucon,Lvar,Uvar = None,None,None,None,None,None,None,None
    n,p,m = 0,0,0
    #print os.listdir(path)
    for file in os.listdir(path):
        if file == 'Q.txt':
            Q =  (np.genfromtxt(path +'Q.txt'))
            p, n = Q.shape
        elif file == 'B.txt':
            B =  (np.genfromtxt(path +'B.txt'))
        elif file == 'd.txt':
            d =  (np.genfromtxt(path +'d.txt'))
        elif file == 'c.txt':
            c =  (np.genfromtxt(path +'c.txt'))
        elif file == 'Lcon.txt':
            Lcon =  (np.genfromtxt(path +'Lcon.txt'))
        elif file == 'Ucon.txt':
                Ucon =  (np.genfromtxt(path +'Ucon.txt'))
        elif file == 'Lvar.txt':
                    Lvar = as_llmatn(p.genfromtxt(path +'Lvar.txt'))
        elif file == 'Uvar.txt':
                    Uvar =  (np.genfromtxt(path +'Uvar.txt'))
        elif file == 'delta.txt':
            delta =  (np.genfromtxt(path +'delta.txt'))
        else:
            print "name must be one of the following names:\n"\
                  +"Q,B,d,c,Lcon,Ucon,Lvar,Uvar"

    
    if B is None:
        m=n
        B = np.zeros([m,n])
    if c is None:
        c = np.zeros(n) 
    if Lcon is None:
        Lcon = -np.ones(m)*inf
    if Ucon is None:
        Ucon = np.ones(m)*inf
    if Lvar is None:
        Lvar = -np.ones(n)*inf
    if Uvar is None:
        Uvar = np.ones(n)*inf
    print path
    name = str(n)+'_'+str(p)+'_'+str(m)+'_l1_tp'+'.txt'
    return Q, B, d, c, Lcon, Ucon, Lvar, Uvar, delta, name

def convert_txt_to_npz():
    from os import walk
    path = "/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/"
    l = list(walk(path))
    dirs = l[0][1][:-1]
    for dir in dirs:
	#print dirs,'hhh'
	Q,B,d,c,lcon,ucon,lvar,uvar,name =\
	 fourth_class_tp(path= path+str(dir)+'/')
	lsqpr = lsq_tp_generator(Q,B,d,c,lcon,ucon,lvar,uvar,name,Model=LSQRModel)
        print lsqpr.name[:-3]

def nnz_elements(probname='None',regqp='None'):
    #path = '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/'
    #strname = probname
    #namef = None
    #list = strname.split('_')
    #p = int(list[0])
    #n = int(list[1])/2
    #os.chdir(path)
    #a =[]
    #for r,d,f in os.walk(path):
	    #a.append(d)
    #tmp = str(p)+'_'+str(n)
    #for i in a[0]:
	#lis = i.split('_')
	#if lis[0] !='tmp':
	    #if  tmp in str(lis[0])+'_'+str(int(lis[1])/2):
		#namef = i
    #tol = 1e-4
    
    #if   namef  :
        #x0 = (numpy.genfromtxt(path+ namef +'/x0.txt'))
        #xmax0 = tol*numpy.linalg.norm(x0,numpy.inf)
        #nnz0 = len(x0[abs(x0)>xmax0])*1.0/len(x0)*100.
        #x = regqp[:n]
        #x_x0 = x-x0
        #x = regqp[:n]
        #xmax = tol*numpy.linalg.norm(x,numpy.inf)  
        #nnz = len(x[x> xmax])*1.0/len(x)*100
    ##for i in range(n):
	    ##print '%9.4f' % x[i]
    #else:
    zero = numpy.zeros((5,), dtype=numpy.float)
    x0 = zero; x = zero; nnz = 0; nnz0 =0
    return x0,x,nnz,nnz0

def read_ampl(path='/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/AMPL/', name=None):
    
    format = 'a+'
    if os.path.exists(path+'output.py'):
        os.remove(path+'output.py')
	
    if os.path.exists(path+'results.py'):
        os.remove(path+'results.py')
    fwrite = open(path+'results.py', format)
    list =['Solve', 'CPLEX 11.2.0: optimal solution;', 'separable QP', # CPLEX
           'Number of Iterations....:',# IPOPT
           'Total CPU secs in IPOPT',           
           'Objective...............:',
           'Dual infeasibility......:',
           'Constraint violation....:',
           'Total CPU secs in IPOPT (w/o function evaluations)   =',
           'PF DF',#LOQO
           'LOQO 6.01: optimal solution',
           'primal objective',
           'Final objective value',#KNITRO
           'Final feasibility error (abs / rel)',
           'Final optimality error  (abs / rel)',
           '# of iterations', 
           'Time spent in evaluations (secs)',
           'Max Primal infeas',  #MINOS,SNOPT
           'Max Dual infeas',
           'solve:',
           #'excluding minos setup:',
           'iterations, objective'
           ]
    name = str(name)
    list_file=['CPLEX_'+name, 'IPOPT_'+name, 'KNITRO_'+name, 'LOQO_'+name,\
               'MINOS_'+name, 'SNOPT_'+name ]
    j = 0
    fread = open(path+'LOQO_'+name,'r')
    for line in fread:
	if 'PF DF' in line : j +=1
	
    for file in list_file:
	fread = open(path+file,'r')
	tmp_line = ''
	for line in fread:
	    for item in list:
		#print item
		if  item in line and tmp_line is not line:
		    fwrite.write(line)
		    tmp_line = line
    fwrite.close()
    
    
    #read = open(path+'results.py','r')
    #jn=0;jm=0;jm1=1
    #for line in read:
	#if 'separable QP barrier' in line :
	    #jn = 1
        #if 'CPLEX 11.2.0:' in line:
	    #jm = 1
	#if 'unbounded problem.' in line :
	    #jm1 = 0

    #if jn ==0:
	#replace_line(path+'results.py', 2, '-1 Number of Iterations....:'+'\n')
    #if jm == 0 or jm1 == 0:
	#replace_line(path+'results.py', 1, 'CPLEX 11.2.0: optimal solution; objective -1'+'\n')
    
    fwrite = open(path+'output.py', format)
    read = open(path+'results.py','r')

    for line in read:
	if 'Max Primal infeas' in line:
	    line = line.replace('Max Primal infeas','PF')
	    fwrite.write(line)
	if 'Max Dual infeas' in line:
	    line = line.replace('Max Dual infeas','DF')
	    fwrite.write(line)
	if 'Final feasibility error (abs / rel)' in line:
	    line = line.replace('Final feasibility error (abs / rel)','DF')
	    fwrite.write(line)
	if 'Final optimality error  (abs / rel)' in line:
	    line = line.replace('Final optimality error  (abs / rel)','PF')
	    fwrite.write(line)
	if 'Dual infeasibility......:' in line:
	    line = line.replace('Dual infeasibility......:','DF')
	    fwrite.write(line)
	if 'Constraint violation....:' in line:
	    line = line.replace('Constraint violation....:','PF')
	    fwrite.write(line)
	if 'PF DF' in line:
	    line.replace('PF DF','PF_DF')
	    fwrite.write(line)
	if 'Objective...............:' in line :
	    line = line.replace('Objective...............:','Objective')
	    fwrite.write(line)
	if 'CPLEX 11.2.0: optimal solution;' in line :
	    line = line.replace('CPLEX 11.2.0: optimal solution;','')
	    fwrite.write(line)
	if 'separable QP barrier' in line :
	    line = line.replace('separable QP barrier','')
	    fwrite.write(line)
	if 'Number of Iterations....:' in line :
	    line = line.replace('Number of Iterations....:',' iterations')
	    fwrite.write(line)

	if 'Total CPU secs in IPOPT (w/o function evaluations)' in line :
	    line = line.replace('Total CPU secs in IPOPT (w/o function evaluations)','Time')
	    fwrite.write(line)
	if 'Solve =' in line :
		    line = line.replace('Solve =','Time')
		    fwrite.write(line)
	if '# of iterations' in line :
		    line = line.replace('# of iterations','iterations')
		    fwrite.write(line)
		
	if 'spent in evaluations (secs)' in line :
		    line = line.replace('spent in evaluations (secs)','')
		    fwrite.write(line)
		
	if 'Final objective value' in line :
		    line = line.replace('Final objective value','objective')
		    fwrite.write(line)
		
	if 'LOQO 6.01: optimal solution (' in line :
		    line = line.replace('LOQO 6.01: optimal solution (',\
		                        ' iterations ')
		    fwrite.write(line)
	#if 'LOQO 6.01: primal and/or dual infeasible'  in line:
	    #line = line.replace('LOQO 6.01: primal and/or dual infeasible',\
		                        #' iterations -1 ')
	    #fwrite.write(line)
	if 'iterations, objective' in line :
		    line = line.replace('iterations, objective',\
		                        'iterations objective')
		    fwrite.write(line)
	if 'primal' in line :
		    line = line.replace('primal','')
		    fwrite.write(line)
	if 'solve:' in line :
		    line = line.replace('solve:','Time')
		    fwrite.write(line)
	#if 'MD' in line:
	    #fwrite.write(line)
	
    dic = []
    fwrite.close()
    read = open(path+'output.py','r') 
    for line in read:
	dic.append(line.split())
	
    remove = ['QP','evaluations)','excluding', 'minos', 'setup:',\
              '=','/','Max', 'Dual', 'infeas']
    for item in remove:
	nested_remove(dic, item)
    a = dic[13:]
    for i in range(j-1):
	a.pop(0)

    list=[['CPLEX', dic[:3]],
          ['IPOPT' , dic[3:8]],
          ['KNITRO',dic[8:13]],
          ['LOQO' , a[:4]],
          ['MINOS' , dic[-7:-4]],
          ['SNOPT' , dic[-3:]]
          ]
    
    #[dt.update({k: int(v)}) for dt in dict for k, v in dt.iteritems(
    total = []
    total.append(list[0][1])
    iter = [total[0][2][1],total[0][2][0]]
    obj = total[0][1]
    p = ['PF','0']
    d = ['DF','0']
    time = total[0][0]
    CPLEX = iter + obj + p + d + time
    
    list.pop(0)
    total =[]
    total.append(list[0][1])
    iter = total[0][0]
    obj = [total[0][1][0],total[0][1][1]]
    time = total[0][4]
    p = [total[0][3][0],max(float(total[0][3][1]),float(total[0][3][2]))]
    d = [total[0][2][0],max(float(total[0][2][1]),float(total[0][2][2]))]
    IPOPT = iter  + obj + p + d + time
    
    list.pop(0)
    total =[]
    total.append(list[0][1])
    iter = total[0][3]
    obj = total[0][0]
    p = [total[0][2][0],total[0][2][1]]
    d = [total[0][1][0],total[0][1][1]] 
    time = total[0][4]
    KNITRO = iter  + obj + p + d + time
    
    list.pop(0)
    total =[]
    total.append(list[0][1])
    obj = total[0][3]
    iter = [total[0][2][0],total[0][2][1]]
    try:
	p = [total[0][0][6],total[0][0][2]]
	d = [total[0][0][7],total[0][0][4]]
	time = total[0][1]
	LOQO =  iter + obj + p + d + time
    except IndexError:
	LOQO = None  

    list.pop(0)
    total =[]
    total.append(list[0][1])
    total[0][2][:2] = total[0][2][1],total[0][2][0]
    iter = [total[0][2][0],total[0][2][1]]
    obj = [total[0][2][2],total[0][2][3]]
    p = [total[0][0][0],total[0][0][2]]
    d = [total[0][0][3],total[0][0][-1]]
    time = [total[0][1][0],total[0][1][2]] 
    MINOS = iter + obj + p + d + time

    list.pop(0)
    total =[]
    total.append(list[0][1])
    iter = [total[0][2][1],total[0][2][0]]
    obj =  [total[0][2][2],total[0][2][3]]
    p = [total[0][0][0],total[0][0][2]]
    d = [total[0][0][3],total[0][0][5]]
    time = total[0][1]
    SNOPT =  iter + obj + p + d + time

    solver_results = [['CPLEX',CPLEX],['IPOPT',IPOPT],\
                      ['KNITRO',KNITRO], ['LOQO',LOQO], ['MINOS',MINOS],
                      ['SNOPT',SNOPT]]
    for item in solver_results:
	for i in range(1,10,2):
	    #print item[0],item[1][i-1],float(item[1][i])
	    item[1][i] = float(item[1][i])
    if not os.path.isfile('/Users/Mohsen/Documents/nlpy_mohsen/lsq/Results.txt'):
	f =open ('/Users/Mohsen/Documents/nlpy_mohsen/lsq/Results.txt','a+')
	path =6*' '+'Name'+ 23*' '+'Iter'+13*' '+'Cost'+8*' '+'RelResidual'\
	     +4*' '+'Time'+4*' '+'Nnz_Orgigin'+4*' '+'Nnz_Minimizer'+2*' '\
	     +'norm(x_x0)'
	f.write(path)
	f.write('\n')
	f.write('-'*(86+18+27+4))
    f =open ('/Users/Mohsen/Documents/nlpy_mohsen/lsq/Results.txt','a+')
    #f.write('\n')
    name = name[:-10]
    #print name
    for item in solver_results:
	Nname = name + '_' + str(item[0]).replace(' ', '*')
	#print len(item),item,name+'_'+str(item)
	RealTol = max( float(item[1][5]), float(item[1][7]))
	#print '%3e        ' % RealTol
	f.write('%-30s      ' % Nname[:30])                # Name of file
	f.write('%-5d       ' % int(item[1][1]))    # Number of iteration
	f.write('%3e   ' % float(item[1][3]))  # cost
	f.write('%3e   ' % RealTol)              # RelResidual
	f.write('%1.3f   ' % float(item[1][9])) # Time
	f.write('\n')
    #f.write('\n')
    f.write('='*(86+18+27+4))
    f.write('\n')
    path_ ='/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/'
    src = path_ + "AMPL/."
    dst = path_ + "archived/" + str(name)+"_archived"+"/"
    ignore_files=ignore_patterns('*.pyc', '.DS_Store')
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore_files)
    import glob
    root = glob.glob(src[:-1]+'*')
    for file in root:
	#print file
	os.remove(file)
    #shutil.copy(path_[:-7] + 'Results.txt', path_)
    #for file in list_file:
	#print file
	#remove_type_file(type='.txt',path= path)
	#remove_type_file(type='.py',path= path)
	
    return dic
    #print solver_results
    #print CPLEX 
    #print IPOPT 
    #print KNITRO
    #print LOQO 
    #print MINOS 
    #print SNOPT 

def matlab_dic(probname):
    f =open ('/Users/Mohsen/Documents/nlpy_mohsen/lsq/Results.txt','a+')
    Matlab_l1_ls = load_as_dict('matlab.txt')
    matlab = Matlab_l1_ls[probname]
    print matlab
    name = probname[:-2]+'_l1-ls'
    f.write('%-30s      ' % name)
    f.write('%-5d       ' % int(matlab[0]))
    f.write('%3e   ' % float(matlab[1]))
    f.write('%3e   ' % float(matlab[2]))
    f.write('%1.3f   ' % float(matlab[3]))
    f.write('%3e   ' % float(matlab[4]))
    f.write('%3e   ' % float(matlab[5]))
    f.write('%3e   ' % float(matlab[6]))
    f.write('\n')
    f.write('-'*(86+18+27+4))
    f.write('\n')
    f.close()
    return

def load_as_dict(filename='matlab.txt'):
    return eval("{" + open('/Users/Mohsen/Documents/nlpy_mohsen/lsq/'+filename).read() + "}")

def nested_remove(L, x):
	if x in L:
	    L.remove(x)
	else:
	    for element in L:
		if type(element) is list:
		    nested_remove(element, x)

def first():
    for i in range(1,2):
	print i
	Q,B,d,c,lcon,ucon,lvar,uvar,name = second_class_tp(10,10)
	#lsqpr = lsq_tp_generator(Q,B,d,c,lcon,ucon,lvar,uvar,name,Model=LSQRModel)
	remove_type_file()
	
	Q,B,d,c,lcon,ucon,lvar,uvar,name = second_class_tp(25*i,3*i,3)
	#Q,B,d,c,lcon,ucon,lvar,uvar,name = first_class_tp(12*i,2*i,3*i)
	lsqpr = lsq_tp_generator(Q,B,d,c,lcon,ucon,lvar,uvar,name,Model=LSQRModel)
	#print lsqpr.name

def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines.insert(line_num,text)
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


def test1():
    #namefile= '1024_1024_1024_l1_tp.txt' 
    #'8_10_10_l1_tp.txt'#'180_240_240_l1_tp.txt'#sys.argv[1:]
    #read_ampl(name=namefile[:])
    ##convert_txt_to_npz()
    #a=load_as_dict()
    #print a.keys(),a.values()
    #read_ampl(name='9_8_8_l1_tp.txt')
    # m,p,n
    #from slack_nlp import SlackFrameworkNLP
    #from slack_nlp import SlackFrameworkNLP as SlackFramework
    #from lsq import lsq
    ##Q,B,d,c,lcon,ucon,lvar,uvar,name = third_class_tp(1222,552,2)
    #Q,B,d,c,lcon,ucon,lvar,uvar,name = first_class_tp(12,1,3)
    #Q,B,d,c,lcon,ucon,lvar,uvar,name = second_class_tp(10,10)
    #Q,B,d,c,lcon,ucon,lvar,uvar,name = mat_py(path= "/Users/Mohsen/Documents/MATLAB/matpydata/")
    #path = '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp'
    #Q,B,d,c,lcon,ucon,lvar,uvar,name = fourth_class_tp()
    #lsqpr = lsq_tp_generator(Q,B,d,c,lcon,ucon,lvar,uvar,name,Model=LSQRModel)
    #print lsqpr.name[:-3]
    #for i in range(1,5):
	#Q,B,d,c,lcon,ucon,lvar,uvar,name = second_class_tp(500*i,100*i)
	#lsqpr = lsq_tp_generator(Q,B,d,c,lcon,ucon,lvar,uvar,name,Model=LSQRModel)
	#remove_type_file()
    
    
    #path_ = '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/tp_pkl/'

    #die = load_dic_npz(path_+name)
    #print name
    #lsq_tp_generator(Q,B,d,c,lcon,ucon,lvar,uvar,name)
    #remove_type_file(type='.npz', path=path_)
    #import scipy.io as sc
    #sc.mmwrite('/tmp/myarray',A)
    #sc.mmread('/tmp/myarray')
    #Q = sc.mmread('m_n_mu12_12_0_0')
        #Q = as_llmat(Q.toarray())
    #A, B, C = matrix('m_n_mu_12_12_0.240718244185_0')
    #print spmatrix.ll_mat_from_mtx('A')


    #qmatrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                 #[12, 12, 14, 19, 26, 58, 60, 55],
                 #[14, 13, 16, 24, 40, 57, 69, 56],
                 #[14, 17, 22, 29, 51, 87, 80, 62],
                 #[18, 22, 37, 56, 68, 109, 103, 77],
                 #[24, 35, 55, 64, 81, 104, 113, 92],
                 #[49, 64, 78, 87, 103, 121, 120, 101],
                 #[72, 92, 95, 98, 112, 100, 103, 99]],np.float)
    #from numpy import linalg as LA
    #print as_llmat(qmatrix)
    #print as_llmat(dctt(qmatrix))
    #a = rog.hilb(5,8)
    #print "The condition number of qmatrix",LA.cond(qmatrix), LA.cond(dctt(qmatrix))
    #print "The condition number of a",LA.cond(a), LA.cond(dctt(a))
    #print as_llmat(a)
    #print as_llmat(dctt(a)) 
    Q,B,d,c,lcon,ucon,lvar,uvar,name = fifth_class_tp(12,10)
    #Q,B,d,c,lcon,ucon,lvar,uvar,name = sixeth_class_tp(128,1024)
    lsqpr = lsq_tp_generator(Q,B,d,c,lcon,ucon,lvar,uvar,name,Model=LSQRModel,\
                             txt='False', npz='True')
    print lsqpr.name
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
    
def exampleliop(m=1,n=1):
        
    #Q,B,d,c,lcon,ucon,lvar,uvar,name = first_class_tp(n,m,m)#(2,2,3 )
    Q,B,d,c,lcon,ucon,lvar,uvar,name = fifth_class_tp(m,n)
    #Q,B,d,c,lcon,ucon,lvar,uvar,name = third_class_tp(nvar=18, prowQ=9, mcon=4 )
    
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
    print as_llmat(FormEntireMatrix(m,n,operator_object))
    return
def l1_ls_class_tp(p=0,n=0):
    Q,y = l1_ls_itre(p,n)
    d = y[:,0]
    
    c = np.zeros(n)
    delta = 0.01
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
                    Uvar=uvar, name='test')
    return lsqpr

if __name__ == "__main__":
    #test1()
    #exampleliop()
    #linearoperator()
    from pykrylov.linop import LinearOperator
    import logging
    import sys
    from numpy.linalg import norm
    exampleliop(m=3,n=2)
    remove_type_file()