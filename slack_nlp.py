"""
A framework for converting a general nonlinear program into a form with
(possibly nonlinear) equality constraints and bounds only, by adding slack
variables.
"""

__docformat__ = 'restructuredtext'

import numpy
import numpy as np
from nlpy.model.nlp import NLPModel
from nlpy.tools import List
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
from nlpy.krylov import SimpleLinearOperator
from nlpy.tools import List

class SlackFrameworkNLP( NLPModel ):
    """
    General framework for converting a nonlinear optimization problem to a
    form using slack variables.

    In the latter problem, the only inequality constraints are bounds on
    the slack variables. The other constraints are (typically) nonlinear
    equalities.

    The order of variables in the transformed problem is as follows:

    1. x, the original problem variables.

    2. sL, the slack variables corresponding to general constraints with
       a lower bound only.

    3. sU, the slack variables corresponding to general constraints with
       an upper bound only.

    4. sR, the slack variables corresponding to general constraints with
       a lower bound and an upper bound.

    This framework initializes the slack variables sL and sU to
    zero by default.

    Note that the slack framework does not update all members of AmplModel,
    such as the index set of constraints with an upper bound, etc., but
    rather performs the evaluations of the constraints for the updated
    model implicitly.

    :parameters:
        :nlp:  Original NLP to transform to a slack form.

    """

    def __init__(self, nlp, **kwargs):

        self.nlp = nlp

        # Save number of variables and constraints prior to transformation
        self.original_n = nlp.n
        self.original_m = nlp.m

        # Number of slacks for the constaints
        nSlacks = nlp.nlowerC + nlp.nupperC + nlp.nrangeC
        self.nSlacks = nSlacks

        # Update effective number of variables and constraints
        n = self.original_n + nSlacks
        m = self.original_m + nlp.nrangeC

        Lvar = -np.infty * np.ones(n)
        Uvar = +np.infty * np.ones(n)
        # Copy orignal bounds
        Lvar[:self.original_n] = nlp.Lvar
        Uvar[:self.original_n] = nlp.Uvar

        # Add bounds corresponding to lower constraints
        bot = self.original_n
        self.sL = range(bot, bot + nlp.nlowerC)
        Lvar[bot:bot+nlp.nlowerC] = nlp.Lcon[nlp.lowerC]

        # Add bounds corresponding to upper constraints
        bot += nlp.nlowerC
        self.sU = range(bot, bot + nlp.nupperC)
        Uvar[bot:bot+nlp.nupperC] = nlp.Ucon[nlp.upperC]

        # Add bounds corresponding to range constraints
        bot += nlp.nupperC
        self.sR = range(bot, bot + nlp.nrangeC)
        Lvar[bot:bot+nlp.nrangeC] = nlp.Lcon[nlp.rangeC]
        Uvar[bot:bot+nlp.nrangeC] = nlp.Ucon[nlp.rangeC]

        # No more inequalities. All constraints are now equal to 0
        Lcon = Ucon = np.zeros(m)

        NLPModel.__init__(self, n=n, m=m, name='Slack-'+nlp.name, Lvar=Lvar, \
                          Uvar=Uvar, Lcon=Lcon, Ucon=Ucon)

        # Redefine primal and dual initial guesses
        self.original_x0 = nlp.x0[:]
        self.x0 = np.zeros(self.n)
        self.x0[:self.original_n] = self.original_x0[:]

        self.original_pi0 = nlp.pi0[:]
        self.pi0 = np.zeros(self.m)
        self.pi0[:self.original_m] = self.original_pi0[:]

        return

    def InitializeSlacks(self, val=0.0, **kwargs):
        """
        Initialize all slack variables to given value. This method may need to
        be overridden.
        """
        self.x0[self.original_n:] = val
        return
    
    def obj(self, x):
            """
            Return the value of the objective function at `x`. This function is
            specialized since the original objective function only depends on a
            subvector of `x`.
            """
            return self.nlp.obj(x[:self.original_n])
        
    def grad(self,x):
        "Evaluate objective gradient at x and returns a Numpy array."
        return self.nlp.grad(x[:self.original_n])

    def cons(self, x):
        """
        Evaluate the vector of general constraints for the modified problem.
        Constraints are stored in the order in which they appear in the
        original problem.
        """
        on = self.original_n; nlp = self.nlp

        equalC = nlp.equalC; lowerC = nlp.lowerC
        upperC = nlp.upperC; rangeC = nlp.rangeC

        c = nlp.cons(x[:on])

        c[equalC] -= nlp.Lcon[equalC]
        c[lowerC] -= x[self.sL]
        c[upperC] -= x[self.sU]
        c[rangeC] -= x[self.sR]

        return c
    
    #def Bounds(self, x):
        #"""
        #Evaluate the vector of equality constraints corresponding to bounds
        #on the variables in the original problem.
        #"""
        #lowerB = self.lowerB ; nlowerB = self.nlowerB
        #upperB = self.upperB ; nupperB = self.nupperB
        #rangeB = self.rangeB ; nrangeB = self.nrangeB

        #n  = self.n ; on = self.original_n
        #mslow = on + nrangeC + self.n_con_low
        #msup  = mslow + self.n_con_up
        #nt = self.original_n + self.n_con_low + self.n_con_up
        #ntlow = nt + self.n_var_low

        #t_low  = x[msup:ntlow]
        #t_up   = x[ntlow:]

        #b = numpy.empty(n + nrangeB)
        #b[:n] = x[:]
        #b[n:] = x[rangeB]

        #b[lowerB] -= self.Lvar[lowerB] ; b[lowerB] -= t_low[:nlowerB]

        #b[upperB] -= self.Uvar[upperB] ; b[upperB] *= -1
        #b[upperB] -= t_up[:nupperB]

        #b[rangeB] -= self.Lvar[rangeB] ; b[rangeB] -= t_low[nlowerB:]
        #b[n:]     -= self.Uvar[rangeB] ; b[n:] *= -1
        #b[n:]     -= t_up[nupperB:]

        #return b
    
    def jprod(self, x, v, **kwargs):
        """
        Evaluate the Jacobian matrix-vector product of all equality
        constraints of the transformed problem with a vector `v` (J(x) v).
        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        nlp = self.nlp ; on = self.original_n
        lowerC = nlp.lowerC ; upperC = nlp.upperC ; rangeC = nlp.rangeC

        p = nlp.jprod(x[:on], v[:on])

        # Insert contribution of slacks on general constraints
        p[lowerC] -= v[self.sL]
        p[upperC] -= v[self.sU]
        p[rangeC] -= v[self.sR]

        return p

    def jtprod(self, x, v, **kwargs):
        """
        Evaluate the Jacobian-transpose matrix-vector product of all equality
        constraints of the transformed problem with a vector `v` (J(x).T v).
        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        nlp = self.nlp
        on = self.original_n ; n = self.n
        lowerC = nlp.lowerC ; upperC = nlp.upperC ; rangeC = nlp.rangeC

        p = np.zeros(n)
        p[:on] = nlp.jtprod(x[:on], v)

        # Insert contribution of slacks on general constraints
        p[self.sL] = -v[lowerC]
        p[self.sU] = -v[upperC]
        p[self.sR] = -v[rangeC]

        return p

    def A(self):
        """
        Return the constraint matrix if the problem is a linear program. See the
        documentation of :meth:`jac` for more information.
        """
        return self.jac([0])

    def hprod(self, x, y, v, **kwargs):
        """
        Evaluate the Hessian vector product of the Lagrangian.
        """
        if y is None: y = np.zeros(self.m)
        # Create some shortcuts for convenience
        nlp = self.nlp ; on = self.original_n

        Hv = np.zeros(self.n)
        Hv[:on] = nlp.hprod(x[:on], y, v[:on], **kwargs)
        return Hv 
    
    def jac(self, x, **kwargs):
        # Create some shortcuts for convenience
        on = self.original_n        
        return SimpleLinearOperator(self.n, self.m, symmetric=False,
                         matvec=lambda u: self.jprod(x,u,**kwargs),
                         matvec_transp=lambda u: self.jtprod(x[:on],u,**kwargs))
    
    def hess(self, x, z=None,u=0, **kwargs):
        """
        Evaluate the Hessian of the Lagrangian.
        """
        if z is None: z = np.zeros(self.m)
        # Create some shortcuts for convenience
        on = self.original_n
        return SimpleLinearOperator(self.n, self.n, symmetric=True,
                         matvec=lambda u: self.hprod(x[:on],z,u,**kwargs))

if __name__ == '__main__':
    from lsqmodel import LSQModel
    #from lsq import lsq
    #from mfnlp import *
    from lsq_testproblem_old import *
    import numpy as np
    n=2;m= 1;
    lsqpr  =  exampleliop(n,m)
    print lsqpr.Q.to_array()
    print lsqpr.B.to_array()
    print lsqpr.Lcon
    print lsqpr.Ucon
    print lsqpr.Lvar
    print lsqpr.Uvar    
    
    
    #lsqp = exampleliop()
    
    slack = SlackFrameworkNLP( lsqpr ) 
    j = slack.jac(slack.x0)

    print 50*"*"
    #print slack.cons(slack.x0).shape
    #print slack.A().shape
    jac = slack.jac(slack.x0)

    n,m = jac.shape
    p, q = lsqpr.jac(0).shape

    print '#'*80
    print ReducedLinearOperator(lsqpr.jac(0), range(0,p),range(0,q)).to_array()
    print '#'*80
    print ReducedLinearOperator(jac, range(0,n),range(0,m)).to_array()
    print slack.Lcon
    print slack.Ucon
    print slack.Lvar 
    print slack.Uvar
    #print '#'*80
    #print ReducedLinearOperator(slack.A(), range(0,n),range(0,m)).to_array()


