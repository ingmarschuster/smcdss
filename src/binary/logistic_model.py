#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from time import clock
from auxpy.data import *
from numpy import *
from scipy.weave import inline, converters
from platform import system
from scipy.linalg import solve

from binary import ProductBinary
from binary.loglinear_model import calc_marginal

CONST_PRECISION = 0.00001
CONST_ITERATIONS = 30


if system() == 'Linux':    hasWeave = True
else:                      hasWeave = False


class LogisticRegrBinary(ProductBinary):
    '''
        A multivariate Bernoulli with conditionals based on logistic regression models.
    '''

    def __init__(self, Beta):
        '''
            Constructor.
            @param Beta matrix of regression coefficients
        '''

        ## matrix of regression coefficients 
        self.Beta = Beta

        ProductBinary.__init__(self, name='logistic-regression-binary', longname='A multivariate Bernoulli with conditionals based on logistic regression models.')

    @classmethod
    def independent(cls, p):
        '''
            Constructs a hidden-normal-binary model with independent components.
            @param cls class 
            @param p mean
        '''
        d = p.shape[0]
        p[1:] = log(p[1:] / (1 - p[1:]))
        Beta = zeros((d, d))
        Beta[:, 0] = p
        return cls(Beta)

    @classmethod
    def uniform(cls, d):
        '''
            Constructs a hidden-normal-binary model with independent components.
            @param cls class 
            @param p mean
        '''
        Beta = zeros((d, d))
        Beta[0, 0] = 0.5
        return cls(Beta)

    @classmethod
    def random(cls, d, scale=3.0):
        '''
            Constructs a random logistic-regression-binary model for testing.
            @param cls class 
            @param d dimension
        '''
        cls = LogisticRegrBinary.independent(random.random(d))
        cls.Beta[:, 1:] = random.normal(scale=scale, size=(d, d - 1))
        return cls

    @classmethod
    def from_data(cls, sample, verbose=False):
        '''
            Construct a logistic-regression binary model from data.
            @param cls class
            @param sample a sample of binary data
        '''
        return cls(calc_Beta(sample, verbose=verbose))


    @classmethod
    def from_loglinear_model(cls, llmodel):
        '''
            Constructs an approximate logistic-regression model from a log-linear model.
            @param cls class 
            @param llmodel log-linear model
        '''
        d = llmodel.d
        Beta = zeros((d, d))
        Beta[0, 0] = llmodel.p_0

        A = copy(llmodel.A)
        for i in xrange(d - 1, 0, -1):
            Beta[i, 1:i + 1] = A[i, :i]
            Beta[i, 1:i + 1] *= 2.0
            Beta[i, 0] = A[i, i]
            A, logc = calc_marginal(A)

        return cls(Beta)

    def _pmf(self, gamma):
        '''
            Probability mass function.
            @param gamma: binary vector
        '''
        return exp(self._lpmf(gamma))

    def _lpmf(self, gamma):
        '''
            Log-probability mass function.
            @param gamma binary vector    
        '''
        if hasWeave:
            return self.__lpmf_weave(gamma)
        else:
            return self.__lpmf_python(gamma)

    def _rvs(self):
        '''
            Samples from the model.
            @return random variable
        '''
        if hasWeave:
            return self.__rvs_weave()[0]
        else:
            return self.__rvs_python()[0]

    def _rvslpmf(self):
        '''
            Generates a random variable and computes its probability.
            @return random variable
            @return likelihood
        '''
        if hasWeave:
            return self.__rvs_weave()
        else:
            return self.__rvs_python()

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.Beta.shape[0]

    def getModelSize(self):
        '''
            Get ratio of used parameters over d*(d+1)/2.
            @return dimension 
        '''
        return '%i/%i' % ((self.Beta <> 0.0).sum(), self.d * (self.d + 1) / 2.0)

    def __str__(self):
        return format_matrix(self.Beta, 'Beta')

    def __lpmf_weave(self, gamma):
        Beta = self.Beta
        d = Beta.shape[0]
        logvprob = empty(1, dtype=float)
        code = \
        """
        double sum, logcprob;
        int i,j;
               
        if (gamma(0)) logvprob = log(Beta(0,0));
        else logvprob = log(1-Beta(0,0));
        
        for(i=1; i<d; i++){
        
            /* Compute log conditional probability that gamma(i) is one */
            sum = Beta(i,0);
            for(j=1; j<=i; j++){        
                sum += Beta(i,j) * gamma(j-1);
            }
            logcprob = -log(1+exp(-sum));
            
            /* Compute log conditional probability of whole gamma vector */
            logvprob += logcprob;        
            if (!gamma(i)) logvprob -= sum;
            
        }
        """
        inline(code, ['d', 'Beta', 'gamma', 'logvprob'], \
                     type_converters=converters.blitz, compiler='gcc')
        return float(logvprob)

    def __lpmf_python(self, gamma):
        Beta = self.Beta
        d = Beta.shape[0]

        if gamma[0]: logvprob = log(Beta[0][0])
        else: logvprob = log(1 - Beta[0][0])

        # Compute log conditional probability that gamma(i) is one for i > 0
        sum = Beta[1:, 0].copy()
        for i in range(1, d): sum[i - 1] += dot(Beta[i, 1:i + 1], gamma[0:i])
        logcprob = -log(1 + exp(-sum))

        # Compute log conditional probability of whole gamma vector
        logvprob += logcprob.sum() - sum[-gamma[1:]].sum()
        return logvprob

    def __rvs_weave(self):
        Beta = self.Beta
        d = Beta.shape[0]
        u = random.rand(d)
        gamma = empty(d, dtype=bool)
        logvprob = empty(1, dtype=float)
        code = \
        """
        double sum, logcprob;
        int i,j;
        
        /* Draw an independent gamma(0) */
        gamma(0) = (u(0) < Beta(0,0));
        
        if (gamma(0)) logvprob = log(Beta(0,0));
        else logvprob = log(1-Beta(0,0));
        
        for(i=1; i<d; i++){
        
            /* Compute log conditional probability that gamma(i) is one */
            sum = Beta(i,0);
            for(j=1; j<=i; j++){        
                sum += Beta(i,j) * gamma(j-1);
            }
            logcprob = -log(1+exp(-sum));
            
            /* Generate the ith entry */
            gamma(i) = (log(u(i)) < logcprob);
            
            /* Compute log conditional probability of whole gamma vector */
            logvprob += logcprob;        
            if (!gamma(i)) logvprob -= sum;
            
        }
        """
        inline(code, ['d', 'u', 'Beta', 'gamma', 'logvprob'], \
                     type_converters=converters.blitz, compiler='gcc')
        return gamma, logvprob

    def __rvs_python(self):
        Beta = self.Beta
        d = Beta.shape[0]
        gamma = empty(d, dtype=bool)
        logu = log(random.rand(d))

        # Draw an independent gamma[0]
        gamma[0] = random.rand() < Beta[0][0]
        if gamma[0]: logvprob = log(Beta[0][0])
        else: logvprob = log(1 - Beta[0][0])

        for i in range(1, d):
            # Compute log conditional probability that gamma(i) is one
            sum = Beta[i][0] + dot(Beta[i, 1:i + 1], gamma[0:i])
            logcprob = -log(1 + exp(-sum))

            # Generate the ith entry
            gamma[i] = logu[i] < logcprob

            # Compute log conditional probability of whole gamma vector
            logvprob += logcprob
            if not gamma[i]: logvprob -= sum

        return gamma, logvprob

    d = property(fget=getD, doc="dimension")



def calc_Beta(sample, Init=None, verbose=False):
    '''
        Computes the logistic regression coefficients of all conditionals. 
        @param sample binary data
        @param Init matrix with inital values
        @param verbose print to stdout 
        @return matrix of regression coefficients
    '''

    if sample.d == 0: return array([])
    if verbose: print 'Computing logistic-regression model of size %i...' % sample.d

    t = clock()
    n = sample.size
    d = sample.d

    A = column_stack((ones(d, dtype=bool)[:, newaxis], abs(sample.getCor(weight=True)) > 0.1))

    # Add constant column.
    X = column_stack((ones(n, dtype=bool)[:, newaxis], sample.proc_data(dtype=bool)))
    if sample.isWeighted: XW = sample.w[:, newaxis] * X
    else: XW = X

    p = X[:, 1:].sum(axis=0) / float(n)
    log_odds = log(p / (1 - p))

    if Init is None:
        Init = zeros((d, d), dtype=float)
        Init[1:, 0] = log_odds[1:]

    Beta = zeros((d, d), dtype=float)
    Beta[0][0] = p[0]

    # Loop over all dimensions compute logistic regressions.
    resp = array([0, 0])
    for m in range(1, d):

        a = where(A[m, :m + 1])[0]
        if a.shape[0] > 1:
            Beta[m, a], r = calc_log_regr(y=X[:, m + 1], X=X[:, a], XW=XW[:, a], init=Init[m, a])
            resp += r
        else:
            Beta[m, 0] = log_odds[m]

    if verbose: print 'Loops %.3f, failures %i, time %.3f\n' % (resp[0] / float(d - 1), resp[1], clock() - t)

    return Beta

def calc_log_regr(y, X, XW=None, init=None):
    '''
        Computes the logistic regression coefficients.. 
        @param y explained variable
        @param X covariables
        @param X weighted covariables
        @param init initial value
        @return vector of regression coefficients
    '''
    n = X.shape[0]
    d = X.shape[1]

    if init is None: beta = zeros(d)
    else:            beta = init

    for iter in range(CONST_ITERATIONS):

        last_beta = beta.copy()

        if hasWeave:
            v = empty(n)
            P = empty(n);
            code = \
            """
            double p, Xbeta;
            
            for (int i = 0; i < n; i++)
            {
                Xbeta = 0;
                for(int k = 0; k <= d; k++)
                {
                    Xbeta += X(i,k) * beta(k);
                }
                p = 1 / (1 + exp(-Xbeta));
                P(i) = p * (1-p);
                v(i) = P(i) * Xbeta + y(i) - p;
            }
            """
            inline(code, ['beta', 'X', 'y', 'P', 'd', 'n', 'v'], \
            type_converters=converters.blitz, compiler='gcc')
        else:
            Xbeta = dot(X, beta)
            p = pow(1 + exp(-Xbeta), -1)
            P = p * (1 - p)
            v = P * Xbeta + y - p

        XWDX = dot(XW.T, P[:, newaxis] * X) + exp(-10) * eye(d)

        # Solve Newton-Raphson equation.
        beta = solve(XWDX, dot(XW.T, v), sym_pos=True)

        if (abs(last_beta - beta) < CONST_PRECISION).all(): break

    # Mark as failed, if the Newton-Raphson iteration did not converge.
    failed = (iter == CONST_ITERATIONS)
    if failed:
        beta = zeros(d)
        p = y.sum() / float(n)
        beta[0] = log(p / (1 - p))

    return beta, (iter, failed)
