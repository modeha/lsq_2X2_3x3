from dctt import partial_DCT,DCT
from lsq_testproblem_old import *
from lsqmodel import LSQModel, LSQRModel
from nlpy import __version__
from slack_nlp import SlackFrameworkNLP as SlackFramework
from lsqp import RegLSQPInteriorPointSolver
from lsqp import RegLSQPInteriorPointSolver3x3
from lsqp import RegLSQPInteriorPointSolver4x4
from lsqp import RegLSQPInteriorPointIterativeSolver4x4
from nlpy.tools.norms import norm2
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
import os
import sys
import logging

# Create root logger.
log = logging.getLogger('cqp')
log.setLevel(logging.INFO)
fmt = logging.Formatter('%(name)-10s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

# Configure the solver logger.
sublogger = logging.getLogger('cqp.solver')
sublogger.setLevel(logging.INFO)
sublogger.addHandler(hndlr)
sublogger.propagate = False

usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent convex quadratic programs."""

# Define formats for output table.
hdrfmt = '%-15s  %5s  %15s  %7s  %7s  %7s  %6s  %6s  %4s'
hdr = hdrfmt % ('Name', 'Iter', 'Objective', 'pResid', 'dResid',
                'Gap', 'Setup', 'Solve', 'Stat')
fmt = '%-15s  %5d  %15.8e  %7.1e  %7.1e  %7.1e  %6.2f  %6.2f  %4s'

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

# File name options
parser.add_option("-i", "--iter", action="store", type="int", default=None,
        dest="maxiter",  help="Specify maximum number of iterations")
parser.add_option("-t", "--tol", action="store", type="float", default=None,
        dest="tol", help="Specify relative stopping tolerance")

parser.add_option("-n", "--n_dim", action="store", type="int", default=None,
        dest="n_dim", help="Specify relative stopping tolerance")

parser.add_option("-m", "--m_dim", action="store", type="int", default=None,
        dest="m_dim", help="Specify relative stopping tolerance")

parser.add_option("-r", "--delta_size", action="store", type="float", default=None,
        dest="delta_size", help="Specify relative stopping tolerance")


parser.add_option("-p", "--regpr", action="store", type="float", default=None,
        dest="regpr", help="Specify initial primal regularization parameter")
parser.add_option("-d", "--regdu", action="store", type="float", default=None,
        dest="regdu", help="Specify initial dual regularization parameter")
 
parser.add_option("-l", "--long-step", action="store_true", default=False,
        dest="longstep", help="Use long-step method")
parser.add_option("-f", "--assume-feasible", action="store_true",
        default=False, dest="assume_feasible",
        help="Deactivate infeasibility check")
parser.add_option("-V", "--verbose", action="store_true", default=False,
        dest="verbose", help="Set verbose mode")

# Parse command-line options
(options, args) = parser.parse_args()
Probname = ''
print 'Brave man! Using example block system!'
Probname +='_4x4'
Solver = RegLSQPInteriorPointIterativeSolver4x4   # RegLSQInteriorPointSolver4x4

opts_init = {}
if options.regpr is not None:
    opts_init['regpr'] = options.regpr
if options.regdu is not None:
    opts_init['regdu'] = options.regdu
    
opts_solve = {}
if options.maxiter is not None:
    opts_solve['itermax'] = options.maxiter
if options.tol is not None:
    opts_solve['tolerance'] = options.tol

if options.m_dim is not None:
    m = options.m_dim
    
if options.n_dim is not None:
    n = options.n_dim

if options.delta_size is not None:
    delta = options.delta_size
# Set printing standards for arrays.
numpy.set_printoptions(precision=3, linewidth=70, threshold=10, edgeitems=2)

multiple_problems = len(args) > 1

#if not options.verbose:
    #log.info(hdr)
    #log.info('-'*len(hdr))
args = ['example']    
for probname in args:
    t_setup = cputime()
    #lsqp = DCT(n,m,delta)#partial_
    #n=2;m= 2;
    #lsqp= exampleliop(n,m)
    lsqp = partial_DCT(n,m,delta)
	
    t_setup = cputime() - t_setup

    # Pass problem to RegQP.
    regqp = Solver(lsqp,
                   verbose=options.verbose,
                   **opts_init)

    regqp.solve(PredictorCorrector=not options.longstep,
                check_infeasible=not options.assume_feasible,
                **opts_solve)

    # Display summary line.
    probname=os.path.basename(probname)
    if not options.verbose:

        sys.stdout.write(fmt % (probname, regqp.iter, regqp.obj_value,
                                regqp.pResid, regqp.dResid, regqp.rgap,
                                t_setup, regqp.solve_time, regqp.short_status))
        if regqp.short_status == 'degn':
            sys.stdout.write(' F')  # Could not regularize sufficiently.
        sys.stdout.write('\n')
    lsqp.close()

log.info('-'*len(hdr))

if not multiple_problems:
    ##    _folder = str(os.getcwd()) + '/binary/'
##    x0 = numpy.load(_folder+str(m)+'_'+str(n)+'.npz')
##    x0 = x0[x0.files[0]]
##    nnz0 = nnz_elements(x0,1e-3)
##    log.info('Non zero elements in original signal %6.f'%nnz0)
    x = regqp.x[0:n]

    nnz = nnz_elements(x,1e-3)
    numpy.set_printoptions(threshold=5)
    numpy.set_printoptions(threshold='nan')
    #print x
    
   
    log.info('Problem name %10s'%probname[:-6])
    #log.info('Non zero elements in minimizer %6.f'%nnz)


    # log.info('Final x: %s, |x| = %7.1e' % (repr(regqp.x),norm2(regqp.x)))
    # log.info('Final y: %s, |y| = %7.1e' % (repr(regqp.y),norm2(regqp.y)))
    # log.info('Final z: %s, |z| = %7.1e' % (repr(regqp.z),norm2(regqp.z)))

    log.info(regqp.status)
    log.info('#Iterations: %-d' % regqp.iter)
    log.info('RelResidual: %7.1e' % regqp.kktResid)
    log.info('Final cost : %21.15e' % regqp.obj_value)
    log.info('Setup time : %6.2fs' % t_setup)
    log.info('Solve time : %6.2fs' % regqp.solve_time)
type='.pyc'; path = os.getcwd()
for file in os.listdir(path):
    if str(file[-len(type):]) ==type:
        os.remove(path+'/'+file)