from lsq_testproblem import *
import sys
if __name__ == "__main__":
    if len(sys.argv)==3:
        namefile= sys.argv[1]
        #print namefile[:-10]
        matlab_dic(namefile[:-10])
    elif len(sys.argv)==2:
        namefile= sys.argv[1]
        read_ampl(name=namefile)
    else:
        path = '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/tmp'
        Q,B,d,c,lcon,ucon,lvar,uvar,name = fourth_class_tp()
        lsqpr = lsq_tp_generator(Q,B,d,c,lcon,ucon,lvar,uvar,name,Model=LSQRModel)
        #print lsqpr.name[:-3]
    remove_type_file()