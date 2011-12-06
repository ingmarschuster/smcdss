#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with logistic conditionals. """

"""
\namespace binary.logistic_cond
$Author: christian.a.schafer@gmail.com $
$Rev: 159 $
$Date: 2011-11-03 11:42:31 +0100 (jeu., 03 nov. 2011) $
@details
"""

import numpy
cimport numpy

cdef extern from "math.h":
    double exp(double)
    double log(double)

import scipy.linalg
import sys
import time
import utils

import binary.product
import binary.qu_exponential
import binary.wrapper

class LogisticCondBinary(binary.product.ProductBinary):
    """ Binary parametric family with logistic conditionals. """

    @staticmethod
    def logistic(x): return 1.0 / (1.0 + numpy.exp(-x))

    def __init__(self, Beta, py_wrapper=None, name='logistic conditionals family', long_name=__doc__):
        """ 
            Constructor.
            \param Beta Lower triangular matrix holding regression coefficients
            \param name name
            \param long_name long name
        """

        logistic = lambda x: 1.0 / (1.0 + numpy.exp(-x))
        p = logistic(numpy.diagonal(Beta))

        # link to python wrapper
        if py_wrapper is None: py_wrapper = binary.wrapper.logistic_cond()

        # call super constructor
        binary.product.ProductBinary.__init__(self, p=p, py_wrapper=py_wrapper, name=name, long_name=long_name)

        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg', 'binary.logistic_cond',)

        # add dependent functions
        self.pp_depfuncs += ('_logistic_cond_all',)

        self.param.update({'Beta':Beta})

    def __str__(self):
        return 'Beta:\n' + repr(self.param['Beta'])

    @classmethod
    def _logistic_cond_all(cls,
                           numpy.ndarray[dtype=numpy.float64_t, ndim=2] Beta,
                           numpy.ndarray[dtype=numpy.float64_t, ndim=2] U=None,
                           numpy.ndarray[dtype=numpy.int8_t, ndim=2] gamma=None):
        """
            All-purpose routine for sampling and point-wise evaluation.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        cdef int d = Beta.shape[0]
        cdef int k, i, size
        cdef double logp = 0.0
        cdef double sum
    
        if U is not None:
            size = U.shape[0]
            gamma = numpy.empty((size, U.shape[1]), dtype=numpy.int8)
    
        if gamma is not None:
            size = gamma.shape[0]
    
        L = numpy.zeros(size, dtype=numpy.float64)
    
        for k in xrange(size):
    
            for i in xrange(0, d):
                # Compute log conditional probability that gamma(i) is one
                sum = Beta[i, i]
                for j in xrange(i):
                    sum += Beta[i, j] * gamma[k, j]
                logcprob = -log(1 + exp(-sum))
    
                # Generate the ith entry
                if U is not None: gamma[k, i] = log(U[k, i]) < logcprob
    
                # Add to log conditional probability
                L[k] += logcprob
                if not gamma[k, i]: L[k] -= sum
    
        return numpy.array(gamma, dtype=bool), L

    @classmethod
    def independent(cls, p):
        """
            Constructs a logistic binary model with independent components.
            \param cls instance
            \param p mean
            \return logistic model
        """
        Beta = numpy.diag(numpy.log(p / (1 - p)))
        return cls(Beta)

    @classmethod
    def uniform(cls, d):
        """ 
            Constructs a uniform logistic binary model.
            \param cls instance
            \param d dimension
            \return logistic model
        """
        Beta = numpy.zeros((d, d))
        return cls(Beta)

    @classmethod
    def random(cls, d, dep=3.0):
        """ 
            Constructs a random logistic binary model.
            \param cls instance
            \param d dimension
            \param dep strength of dependencies [0,inf)
            \return logistic model
        """
        cls = LogisticCondBinary.independent(p=numpy.random.random(d))
        Beta = numpy.random.normal(scale=dep, size=(d, d))
        Beta *= numpy.dot(Beta, Beta)
        for i in xrange(d): Beta[i, i] = cls.param['Beta'][i, i]
        cls.param['Beta'] = Beta
        return cls

    @classmethod
    def from_data(cls, sample, Init=None, job_server=None, eps=0.02, delta=0.075, verbose=False):
        """ 
            Construct a logistic-regression binary model from data.
            \param cls instance
            \param sample a sample of binary data
            \param Init matrix with inital values
            \param eps marginal probs in [eps,1-eps] > logistic model
            \param delta abs correlation in  [delta,1] > association
            \param xi marginal probs in [xi,1-xi] > random component
            \param verbose detailed output
            \return logistic model
        """
        return cls(calc_Beta(sample, Init=Init, job_server=job_server, eps=eps, delta=delta, verbose=verbose))

    def renew_from_data(self, sample, job_server=None, eps=0.02, delta=0.075, lag=0, verbose=False):
        """ 
            Construct a logistic-regression binary model from data.
            \param sample a sample of binary data
            \param eps marginal probs in [eps,1-eps] > logistic model
            \param delta abs correlation in  [delta,1] > association
            \param xi marginal probs in [xi,1-xi] > random component
            \param verbose detailed output
            \return logistic model
        """

        # Compute new parameter from data.
        newBeta = calc_Beta(sample=sample, Init=self.param['Beta'], \
                            job_server=job_server, eps=eps, delta=delta, verbose=verbose)

        # Set convex combination of old and new parameter.
        self.param['Beta'] = (1 - lag) * newBeta + lag * self.param['Beta']
        self.param['p'] = utils.inv_logit(self.Beta.sum(axis=1))

    @classmethod
    def from_loglinear_model(cls, llmodel):
        """ 
            Constructs a logistic model that approximates a log-linear model.
            \param cls instance
            \param llmodel log-linear model
            \todo Instead of margining out in arbitrary order, we could use a greedy approach
            which picks the next dimension by minimizing the error made in the Taylor approximation. 
        """
        d = llmodel.d
        Beta = numpy.zeros((d, d))
        Beta[0, 0] = utils.logit(llmodel.p_0)

        A = numpy.copy(llmodel.A)
        for i in xrange(d - 1, 0, -1):
            Beta[i, 0:i] = A[i, :i] * 2.0
            Beta[i, i] = A[i, i]
            A = binary.qu_exponential.calc_marginal(A)

        return cls(Beta)

    def _getD(self):
        """ Get dimension. \return dimension """
        return self.Beta.shape[0]

    def getBeta(self):
        """ Get Beta matrix. \return Beta-matrix """
        return self.param['Beta']

    Beta = property(fget=getBeta, doc="Beta")


def calc_Beta(sample, eps=0.02, delta=0.05, Init=None, job_server=None, verbose=True):
    """ 
        Computes the logistic regression coefficients of all conditionals. 
        \param sample binary data
        \param eps marginal probs in [eps,1-eps] > logistic model
        \param delta abs correlation in  [delta,1] > association
        \param Init matrix with initial values
        \param job_server job server
        \param verbose print to stdout 
        \return matrix of regression coefficients
    """

    if sample.d == 0: return numpy.array(Init)

    t = time.time()
    n = sample.size
    d = sample.d

    # Avoid numpy.column_stack for it causes a MemoryError    
    X = numpy.empty((n, d + 1))
    X[:n, :d] = sample.proc_data(dtype=float)
    X[:n, d] = numpy.ones(n, dtype=float)

    # Compute weighted sample.
    w = numpy.array(sample.nW)
    XW = w[:, numpy.newaxis] * X

    # Compute slightly adjusted mean and real log odds.
    p = LogisticCondBinary.MIN_MARGINAL_PROB * 0.5 + (1.0 - LogisticCondBinary.MIN_MARGINAL_PROB) * XW[:, 0:d].sum(axis=0)

    # Compute logits.
    logit_p = utils.logit(p)

    # check whether data is sufficient to fit logistic model
    if n == 1: return numpy.diag(logit_p)

    # Find strong associations.    
    L = abs(utils.data.calc_cor(X[:, 0:d], w=w)) > delta

    # Remove components with extreme marginals.
    for i, m in enumerate(p):
        if m < eps or m > 1.0 - eps:
            L[i, :] = numpy.zeros(d, dtype=bool)

    # Initialize Beta with logits on diagonal.
    Beta = numpy.zeros((d, d))
    if Init is None: Init = numpy.diag(logit_p)

    if verbose: stats = dict(regressions=0.0, failures=0, iterations=0, product=d, logistic=0)

    # Loop over all dimensions compute logistic regressions.
    jobs = list()
    if not job_server is None: ncpus = job_server.get_ncpus()

    for i in range(d):
        covariates = list(numpy.where(L[i, 0:i])[0])
        if len(covariates) > 0:
            if not job_server is None:
                jobs.append([i, covariates + [i],
                        (job_server.submit(
                         func=calc_log_regr,
                         args=(X[:, i], X[:, covariates + [d]], XW[:, covariates + [d]], Init[i, covariates + [i]], w, False),
                         modules=('numpy', 'scipy.linalg', 'binary')))
                ])
                # once jobs are assigned to all cpus let the job server
                # wait in order to prevent memory errors on large problems
                ncpus -= 1
                if ncpus <= 0:
                    job_server.wait()
                    ncpus = job_server.get_ncpus()
            else:
                jobs.append([i, covariates + [i], (calc_log_regr(
                              X[:, i], X[:, covariates + [d]], XW[:, covariates + [d]], Init[i, covariates + [i]], w, False))
                ])
        else:
            Beta[i, i] = logit_p[i]

    # wait and retrieve results
    if not job_server is None: job_server.wait()

    # write results to Beta matrix
    for i, covariates, job in jobs:
        if isinstance(job, tuple): beta, iterations = job
        else: beta, iterations = job()

        if verbose:
            stats['iterations'] = iterations
            stats['regressions'] += 1.0
            stats['product'] -= 1
            stats['logistic'] += 1

        if not beta is None:
            Beta[i, covariates] = beta
        else:
            if verbose: stats['failures'] += 1
            Beta[i, i] = logit_p[i]

    if verbose:
        stats.update({'time':time.time() - t})
        if stats['regressions'] > 0: stats['iterations'] /= stats['regressions']
        print 'Logistic model: (p %(product)i, l %(logistic)i), loops %(iterations).3f, failures %(failures)i, time %(time).3f\n' % stats

    return Beta

def calc_log_regr(y, X, XW, init, w=None, verbose=False):
    """
        Computes the logistic regression coefficients.
        \param y explained variable
        \param X covariables
        \param XW weighted covariables
        \param init initial value
        \param w weights
        \param verbose verbose
        \return vector of regression coefficients
    """

    PRECISION = 1e-5

    # Initialize variables. 
    n = X.shape[0]
    d = X.shape[1]
    beta = init
    if w is None: w = numpy.ones(n) / float(n)
    v = numpy.empty(n)
    P = numpy.empty(n)
    llh = -numpy.inf
    _lambda = 1e-8

    for i in xrange(50):

        # Save last iterations values.
        last_llh = llh
        last_beta = beta.copy()

        # Compute Fisher information at beta
        Xbeta = numpy.dot(X, beta)
        Xbeta = numpy.minimum(numpy.maximum(Xbeta, -500), 500)

        p = numpy.power(1 + numpy.exp(-Xbeta), -1)
        P = p * (1 - p)
        XWPX = numpy.dot(XW.T, P[:, numpy.newaxis] * X) + _lambda * numpy.eye(d)
        v = P * Xbeta + y - p

        # Solve Newton-Raphson equation.
        try:
            beta = scipy.linalg.solve(XWPX, numpy.dot(XW.T, v), sym_pos=True)
        except:
            if verbose: print '> likelihood not unimodal'
            beta = scipy.linalg.solve(XWPX, numpy.dot(XW.T, v), sym_pos=False)

        # Compute the log-likelihood.
        llh = -0.5 * _lambda * numpy.dot(beta, beta) + (w * (y * Xbeta + numpy.log(1 + numpy.exp(Xbeta)))).sum()

        if abs(beta).max() > 1e4:
            if verbose: print 'convergence failure\n'
            return None, i

        if numpy.allclose(last_beta, beta, rtol=0, atol=PRECISION):
            if verbose: print 'no change in beta\n'
            break

        if verbose: print '%i) log-likelihood %.2f' % (i + 1, llh)
        if abs(last_llh - llh) < PRECISION:
            if verbose: print 'no change in likelihood\n'
            break
    return beta, i + 1

def adjust_Beta(moments):
    """
        Computes the parameter Beta of a logistic conditionals model that
        corresponds to a given cross-moment matrix.
        
        \param moments cross-moments matrix
        \return Beta parameter
        
        \todo Augment data by one dimension instead of using a completely new sample.
        \todo Introduce repair mode lowering the critical correlation.
        \todo Automatic switch to MC with growing iterator i.
    """

    # dimension
    d = moments.shape[0]
    # number of MC samples
    n = 20000

    # Initialize Beta
    Beta = numpy.zeros((d, d), dtype=float)
    Beta[0, 0] = utils.logit(moments[0, 0])

    # Loop over all dimensions
    for i in xrange(1, d):

        # Build lower dimensional model
        l = LogisticCondBinary(Beta=Beta[:i, :i])

        # Draw random states for MC estimate
        if d > 6:
            X = l.rvs(size=n)
            pmf = numpy.ones(n) / float(n)
        else:
            sample = l.marginals()
            X = sample.X
            pmf = numpy.exp(sample.W)
            n = X.shape[0]

        X = numpy.column_stack((X, numpy.ones(n, dtype=float)[:, numpy.newaxis]))

        # Initialize b with zero vector
        b = numpy.zeros(i + 1)
        b[-1] = utils.logit(moments[i, i])

        # Target vector
        tM = moments[i, :i + 1]

        # Newton iteration
        N = 50
        for j in xrange(N):

            b_before = b.copy()

            # Compute MC estimates for expected values
            f = numpy.zeros(i + 1)
            J = numpy.zeros((i + 1, i + 1))
            for v in xrange(n):
                # Compute marginal probability 
                p = max(min(1.0 / (1.0 + numpy.exp(-numpy.dot(X[v], b))), 1.0 - 1e-8), 1e-8)
                f += pmf[v] * p * X[v]
                J += pmf[v] * (p - p * p) * numpy.dot(X[v][:, numpy.newaxis], X[v][numpy.newaxis, :])

            # Subtract non-random parts            
            f -= tM

            # Newton update
            try:
                b = scipy.linalg.solve(J, numpy.dot(J, b) - f, sym_pos=True)
            except numpy.linalg.linalg.LinAlgError:
                b = scipy.linalg.solve(J + numpy.eye(i + 1) * 1e-8, numpy.dot(J, b) - f, sym_pos=False)
            if (numpy.abs(b - b_before) < LogisticCondBinary.PRECISION).all(): break

        if j >= N - 1:
            b = numpy.zeros(i + 1)
            b[-1] = utils.logit(moments[i, i])

        Beta[i, :i + 1] = b
    return Beta


def random_problem(d, eps=0.05, lambda_=0.5):
    """
        Creates a cross-moments matrix that is consistent with the general
        constraints on binary data.
        
        \param d dimension
        \param eps minmum distance to borders of [0,1]
        \param lambda parameter for convex combination of random off-diagonal and independence
        \return M cross-moment matrix
    """
    M = numpy.diag(eps + (1.0 - 2 * eps) * numpy.random.random(d))
    for i in range(d):
        for j in range(i):
            high = min(M[i, i], M[j, j])
            low = max(M[i, i] + M[j, j] - 1.0, 0)
            M[i, j] = (1.0 - lambda_) * (low + numpy.abs(high - low) * numpy.random.random()) + lambda_ * M[i, i] * M[j, j]
            M[j, i] = M[i, j]
    return M

def corr2moments(m, C):
    """
        Converts a mean vector and correlation matrix to the corresponding
        cross-moment matrix.
        
        \param m mean vector.
        \param C correlation matrix
        \return M cross-moment matrix
    """
    var = (m * (1 - m))[:, numpy.newaxis]
    var = numpy.sqrt(numpy.dot(var, var.T))
    m = m[:, numpy.newaxis]
    M = (C * var) + numpy.dot(m, m.T)
    return M

def moments2corr(M):
    """
        Converts a cross-moment matrix to a corresponding pair of mean vector
        and correlation matrix. .
        
        \param M cross-moment matrix
        \return m mean vector.
        \return C correlation matrix
    """
    m = numpy.diag(M)
    var = (m * (1 - m))[:, numpy.newaxis]
    var = numpy.sqrt(numpy.dot(var, var.T))
    m = m[:, numpy.newaxis]
    R = (M - numpy.dot(m, m.T)) / var
    return numpy.diag(M), R

def example():
    d = 4
    m, R = moments2corr(random_problem(d))
    print m
    print R
    Beta = adjust_Beta(corr2moments(m, R))
    l = LogisticCondBinary(Beta)
    print l.marginals()

def main():
    h = LogisticCondBinary.random(5)
    print h.rvstest(200)

if __name__ == "__main__":
    main()
