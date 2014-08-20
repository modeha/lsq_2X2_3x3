from slack_nlp import SlackFrameworkNLP as SlackFramework

from cqp import RegQPInteriorPointSolver
from cqp import RegQPInteriorPointSolver3x3
from cqp import RegLSQInteriorPointSolver4x4
from cqp import RegLSQInteriorPointIterativeSolver4x4


class RegLSQPInteriorPointSolver(RegQPInteriorPointSolver):
    """
        Solve a least squares program of the form::
        
           minimize    c'x + 1/2|Qx-d|^2
           subject to  L <= Ax <= U,                                 (LSQP-FF)
                       l <=  x <= u,
                       
        by converting this problem (First Form) to the special case of QP 
        (Second Form) :: 
        
            minimize c'x +1/2|r|^2
            subject to. [L] <= [A  0][x] <= [U],
                        [d] <= [Q  I][r] <= [d],                     (LSQP-SF)
                        [l] <=       [x] <= [u],
                     -[inf] <=       [r] <= [inf].
                     
        This second form can be wirtten as the slack form::

           minimize    c'x + 1/2|r|^2
           subject to  A1 x + A2 s = b,                                (LSQP)
                       s >= 0,

        The variables [x,r] are the original problem variables and s are slack
        variables. Any least squares program may be converted to the above form
        by instantiation of the `SlackFramework` class. The conversion to the 
        slack formulation is mandatory in this implementation.

        The method is a variant of Mehrotra's predictor-corrector method where
        steps are computed by solving the primal-dual system in augmented form.

        Primal and dual regularization parameters may be specified by the user
        via the opional keyword arguments `regpr` and `regdu`. Both should be
        positive real numbers and should not be "too large". By default they 
        are set to 1.0 and updated at each iteration.

        If `scale` is set to `True`, (LSQP) is scaled automatically prior to
        solution so as to equilibrate the rows and columns of the constraint
        matrix [A1 A2].

        Advantages of this method are that it is not sensitive to dense columns
        in A, no special treatment of the unbounded variables x is required, and
        a sparse symmetric quasi-definite system of equations is solved at each
        iteration. The latter, although indefinite, possesses a Cholesky-like
        factorization. Those properties makes the method typically more robust
        that a standard predictor-corrector implementation and the linear system
        solves are often much faster than in a traditional interior-point method
        in augmented form.

        :keywords:
            :scale: Perform row and column equilibration of the constraint
                    matrix [A1 A2] prior to solution (default: `True`).

            :regpr: Initial value of primal regularization parameter
                    (default: `1.0`).

            :regdu: Initial value of dual regularization parameter
                    (default: `1.0`).

            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

            :logger_name: Name of a logger to control output.

            :verbose: Turn on verbose mode (default `False`).
        """
    def __init__(self, lsqp, *args, **kwargs):
        
        lsqr = lsq(lsqp)              # convert FF to the SF
        lsqpsf = SlackFramework(lsqr)  # convert SF to slack form
        
        super(RegLSQPInteriorPointSolver, self).__init__(lsqpsf, *args, **kwargs)

class RegLSQPInteriorPointSolver3x3(RegQPInteriorPointSolver3x3):
    """
    A variant of the regularized interior-point method based on the 3x3 block
    system instead of the reduced 2x2 block system.
    """
    def __init__(self, lsqp, *args, **kwargs):
        
        lsqr = lsq(lsqp)                      # convert FF to the SF
        lsqpsf = SlackFramework(lsqr)          # convert SF to slack form
        
        super(RegLSQPInteriorPointSolver3x3, self).__init__(lsqpsf, *args,\
                                                            **kwargs)
        
class RegLSQPInteriorPointSolver4x4(RegLSQInteriorPointSolver4x4):
    """
    A variant of the regularized interior-point method based on the 4x4 block
    system instead of the reduced 2x2 or 3x3 blocks system.
    """
    def __init__(self, lsqp, *args, **kwargs):
        
        lsqpsf = SlackFramework(lsqp)          # convert SF to slack form
        
        super(RegLSQPInteriorPointSolver4x4, self).__init__(lsqpsf, *args,\
                                                            **kwargs)

class RegLSQPInteriorPointIterativeSolver4x4(RegLSQInteriorPointIterativeSolver4x4):
    """
    A variant of the regularized interior-point method based on the 4x4 block
    system instead of the reduced 2x2 or 3x3 blocks system using Iterative solvers.
    """
    def __init__(self, lsqp, *args, **kwargs):
        
        lsqpsf = SlackFramework(lsqp)          # convert SF to slack form
        
        super(RegLSQPInteriorPointIterativeSolver4x4, self).__init__(lsqpsf, *args,\
                                                            **kwargs)



