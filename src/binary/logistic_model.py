#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @Author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

import time
import scipy
from numpy import *

import binary
import inspect
import utils

class LogisticBinary(binary.ProductBinary):
    ''' A binary model with conditionals based on logistic regressions. '''

    def __init__(self, Beta, name='logistic binary',
                 longname='A binary model with conditionals based on logistic regressions.'):
        ''' Constructor.
            @param Beta matrix of regression coefficients
        '''

        binary.ProductBinary.__init__(self, name=name, longname=longname)

        if 'cython' in utils.opts:
            self.f_rvslpmf = utils.python.logistic_cython_rvslpmf
            self.f_lpmf = utils.python.logistic_cython_lpmf
            self.f_rvs = utils.python.logistic_cython_rvs
        else:
            self.f_rvslpmf = utils.python.logistic_rvslpmf
            self.f_lpmf = utils.python.logistic_lpmf
            self.f_rvs = utils.python.logistic_rvs

        self.param = dict(Beta=Beta)

    @classmethod
    def independent(cls, p):
        ''' Constructs a logistic binary model with independent components.
            @param cls instance
            @param p mean
            @return logistic model
        '''
        d = p.shape[0]
        Beta = diag(log(p / (1 - p)))
        return cls(Beta)

    @classmethod
    def uniform(cls, d):
        ''' Constructs a uniform logistic binary model.
            @param cls instance
            @param d dimension
            @return logistic model
        '''
        Beta = zeros((d, d))
        return cls(Beta)

    @classmethod
    def random(cls, d, dep=3.0):
        ''' Constructs a random logistic binary model.
            @param cls instance
            @param d dimension
            @param dep strength of dependencies [0,inf)
            @return logistic model
        '''
        cls = LogisticBinary.independent(p=random.random(d))
        Beta = random.normal(scale=dep, size=(d, d))
        Beta *= dot(Beta, Beta)
        for i in xrange(d): Beta[i, i] = cls.param['Beta'][i, i]
        cls.param['Beta'] = Beta
        return cls

    @classmethod
    def from_data(cls, sample, Init=None, eps=0.02, delta=0.075, xi=0, verbose=False):
        ''' Construct a logistic-regression binary model from data.
            @param cls instance
            @param sample a sample of binary data
            @param Init matrix with inital values
            @param eps marginal probs in [eps,1-eps] > logistic model
            @param delta abs correlation in  [delta,1] > association
            @param xi marginal probs in [xi,1-xi] > random component
            @param verbose detailed output
            @return logistic model
        '''
        return cls(calc_Beta(sample, Init=Init, verbose=verbose))

    def renew_from_data(self, sample, eps=0.02, delta=0.075, xi=0, lag=0, verbose=False):
        ''' Construct a logistic-regression binary model from data.
            @param sample a sample of binary data
            @param eps marginal probs in [eps,1-eps] > logistic model
            @param delta abs correlation in  [delta,1] > association
            @param xi marginal probs in [xi,1-xi] > random component
            @param verbose detailed output
            @return logistic model
        '''
        # Compute new parameter from data.
        newBeta = calc_Beta(sample=sample, Init=self.param['Beta'],
                            eps=eps, delta=delta, verbose=verbose)

        # Set convex combination of old and new parameter.
        self.param['Beta'] = (1 - lag) * newBeta + lag * self.param['Beta']

    @classmethod
    def from_loglinear_model(cls, llmodel):
        ''' Constructs a logistic model that approximates a log-linear model.
            @param cls instance
            @param llmodel log-linear model
        '''
        d = llmodel.d
        Beta = zeros((d, d))
        Beta[0, 0] = utils.logit(llmodel.p_0)

        A = copy(llmodel.A)
        for i in xrange(d - 1, 0, -1):
            Beta[i, 0:i] = A[i, :i] * 2.0
            Beta[i, i] = A[i, i]
            A, logc = binary.loglinear_model.calc_marginal(A)

        return cls(Beta)

    def getD(self):
        ''' Get dimension.
            @return dimension 
        '''
        return self.param['Beta'].shape[0]

    def getModelSize(self):
        ''' Get ratio of used parameters over d*(d+1)/2.
            @return dimension 
        '''
        return '%i/%i' % ((self.Beta <> 0.0).sum(), self.d * (self.d + 1) / 2.0)

    def __str__(self):
        return format_matrix(self.Beta, 'Beta')

    d = property(fget=getD, doc="dimension")

def calc_Beta(sample, eps=0.02, delta=0.05, Init=None, verbose=False):
    ''' Computes the logistic regression coefficients of all conditionals. 
        @param sample binary data
        @param eps marginal probs in [eps,1-eps] > logistic model
        @param delta abs correlation in  [delta,1] > association
        @param Init matrix with initial values
        @param verbose print to stdout 
        @return matrix of regression coefficients
    '''

    if sample.d == 0: return array([])

    t = time.clock()
    n = sample.size
    d = sample.d

    X = column_stack((sample.proc_data(dtype=float), ones(n, dtype=float)[:, newaxis]))
    if sample.ess > 0.1: XW = sample.nW[:, newaxis] * X
    else: XW = X

    # Compute slightly adjusted mean and real log odds.
    p = 1e-08 * 0.5 + (1.0 - 1e-08) * X[:, 0:d].sum(axis=0) / float(n)
    logit = utils.logit(p)

    # Find strong associations
    L = abs(sample.cor) > delta

    # Remove components with extreme marginals
    for i, m in enumerate(p):
        if m < eps or m > 1.0 - eps:
            L[i, :] = zeros(d, dtype=bool)

    Beta = diag(logit)
    if Init is None: Init = Beta

    if verbose: stats = dict(regressions=0.0, failures=0, iterations=0,
                             product=d - diag(L).sum(), logistic=diag(L).sum())

    # Loop over all dimensions compute logistic regressions.
    for i in range(1, d):
        l = list(where(L[i, 0:i])[0])
        if len(l) > 0:
            beta, iterations = calc_log_regr(y=X[:, i], X=X[:, l + [d]],
                                             XW=XW[:, l + [d]], init=Init[i, l + [i]])
            if verbose:
                stats['iterations'] = iterations
                stats['regressions'] += 1.0

            # component failed to converge due to complete separation
            if beta is None:
                stats['failures'] += 1
                Beta[i, i] = logit[i]
            else:
                Beta[i, l + [i]] = beta

    if verbose:
        stats.update(dict(product=stats['product'] + stats['failures'],
                          logistic=stats['logistic'] - stats['failures'],
                          time=time.clock() - t))
        if stats['regressions'] > 0: stats['iterations'] /= stats['regressions']
        print 'Logistic model: (p %(product)i, l %(logistic)i), loops %(iterations).3f, failures %(failures)i, time %(time).3f\n' % stats

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

    v = empty(n)
    P = empty(n)

    for iter in range(binary.CONST_ITERATIONS):

        last_beta = beta.copy()

        if False:
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

        #if True: # not hasWeave or P[0] != P[0]:
        #if hasWeave: print '\n\n\nNUMERICAL ERROR USING WEAVE!\n\n\n'

        Xbeta = dot(X, beta)
        p = pow(1 + exp(-Xbeta), -1)
        P = p * (1 - p)
        v = P * Xbeta + y - p

        XWDX = dot(XW.T, P[:, newaxis] * X) + 1e-4 * eye(d)

        # Solve Newton-Raphson equation.
        try:
            beta = scipy.linalg.solve(XWDX, dot(XW.T, v), sym_pos=True)
        except:
            try:
                beta = scipy.linalg.solve(XWDX, dot(XW.T, v), sym_pos=False)
            except:
                print format(XWDX, 'XWDX')
                print format(dot(XW.T, v), 'XW_v')
                raise ValueError

        # convergence failure due to complete separation
        if abs(beta[d - 1]) > 25: return None, iter

        if (abs(last_beta - beta) < binary.CONST_PRECISION).all(): break

    return beta, iter


















#    def __lpmf_weave(self, gamma):
#        Beta = self.Beta
#        d = Beta.shape[0]
#        logvprob = empty(1, dtype=float)
#        code = \
#        """
#        double sum, logcprob;
#        int i,j;
#               
#        if (gamma(0)) logvprob = log(Beta(0,0));
#        else logvprob = log(1-Beta(0,0));
#        
#        for(i=1; i<d; i++){
#        
#            /* Compute log conditional probability that gamma(i) is one */
#            sum = Beta(i,0);
#            for(j=1; j<=i; j++){        
#                sum += Beta(i,j) * gamma(j-1);
#            }
#            logcprob = -log(1+exp(-sum));
#            
#            /* Compute log conditional probability of whole gamma vector */
#            logvprob += logcprob;        
#            if (!gamma(i)) logvprob -= sum;
#            
#        }
#        """
#        inline(code, ['d', 'Beta', 'gamma', 'logvprob'], \
#                     type_converters=converters.blitz, compiler='gcc')
#        return float(logvprob)
#
#    def __lpmf_python(self, gamma):
#        Beta = self.Beta
#        d = Beta.shape[0]
#
#        if gamma[0]: logvprob = log(Beta[0][0])
#        else: logvprob = log(1 - Beta[0][0])
#
#        # Compute log conditional probability that gamma(i) is one for i > 0
#        sum = Beta[1:, 0].copy()
#        for i in range(1, d): sum[i - 1] += dot(Beta[i, 1:i + 1], gamma[0:i])
#        logcprob = -log(1 + exp(-sum))
#
#        # Compute log conditional probability of whole gamma vector
#        logvprob += logcprob.sum() - sum[-gamma[1:]].sum()
#        return logvprob
#
#    def __rvs_weave(self):
#        Beta = self.Beta
#        d = Beta.shape[0]
#        u = random.rand(d)
#        gamma = empty(d, dtype=bool)
#        logvprob = empty(1, dtype=float)
#        code = \
#        """
#        double sum, logcprob;
#        int i,j;
#        
#        /* Draw an independent gamma(0) */
#        gamma(0) = (u(0) < Beta(0,0));
#        
#        if (gamma(0)) logvprob = log(Beta(0,0));
#        else logvprob = log(1-Beta(0,0));
#        
#        for(i=1; i<d; i++){
#        
#            /* Compute log conditional probability that gamma(i) is one */
#            sum = Beta(i,0);
#            for(j=1; j<=i; j++){        
#                sum += Beta(i,j) * gamma(j-1);
#            }
#            logcprob = -log(1+exp(-sum));
#            
#            /* Generate the ith entry */
#            gamma(i) = (log(u(i)) < logcprob);
#            
#            /* Compute log conditional probability of whole gamma vector */
#            logvprob += logcprob;        
#            if (!gamma(i)) logvprob -= sum;
#            
#        }
#        """
#        inline(code, ['d', 'u', 'Beta', 'gamma', 'logvprob'], \
#                     type_converters=converters.blitz, compiler='gcc')
#        return gamma, logvprob
#
#    def __rvs_python(self):
#        Beta = self.Beta
#        d = Beta.shape[0]
#        gamma = empty(d, dtype=bool)
#        logu = log(random.rand(d))
#
#        # Draw an independent gamma[0]
#        gamma[0] = random.rand() < Beta[0][0]
#        if gamma[0]: logvprob = log(Beta[0][0])
#        else: logvprob = log(1 - Beta[0][0])
#
#        for i in range(1, d):
#            # Compute log conditional probability that gamma(i) is one
#            sum = Beta[i][0] + dot(Beta[i, 1:i + 1], gamma[0:i])
#            logcprob = -log(1 + exp(-sum))
#
#            # Generate the ith entry
#            gamma[i] = logu[i] < logcprob
#
#            # Add to log conditional probability
#            logvprob += logcprob
#            if not gamma[i]: logvprob -= sum
#
#        return gamma, logvprob

#        if len(adjIndex) == 0:
#            self.Beta = array([])
#            self.Beta.shape = (0, 0)
#            return
#
#        prvBeta = self.Beta
#        adjBeta = zeros((len(adjIndex), len(adjIndex)), dtype = float)
#
#        mapping = [(adjIndex.index(i), prvIndex.index(i)) for i in list(set(prvIndex) & set(adjIndex))]
#        for adj_x, prv_x in mapping:
#            adjBeta[adj_x, 0] = prvBeta[prv_x, 0]
#            for adj_y, prv_y in mapping:
#                if adj_y >= len(adjIndex) - 1 or prv_y >= len(prvIndex) - 1: continue
#                adjBeta[adj_x, adj_y + 1] = prvBeta[prv_x, prv_y + 1]
