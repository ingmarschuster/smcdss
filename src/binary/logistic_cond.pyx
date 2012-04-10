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
import binary.base
import binary.qu_exponential
import binary.wrapper

class LogisticCondBinary(binary.product.ProductBinary):
    """ Binary parametric family with logistic conditionals. """

    PRECISION = binary.base.BaseBinary.PRECISION
    MAX_ENTRY_SUM = numpy.finfo(float).maxexp * log(2)

    def __init__(self, Beta, name='logistic conditionals family', long_name=__doc__):
        """ 
            Constructor.
            \param Beta Lower triangular matrix holding regression coefficients
            \param name name
            \param long_name long name
        """

        p = logistic(numpy.diagonal(Beta))

        # call super constructor
        binary.product.ProductBinary.__init__(self, p=p, name=name, long_name=long_name)

        self.py_wrapper = binary.wrapper.logistic_cond()

        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg', 'binary.logistic_cond',)

        # add dependent functions
        self.pp_depfuncs += ('_rvslpmf_all',)

        self.Beta = Beta

    def __str__(self):
        return 'd: %d, Beta:\n%s' % (self.d, repr(self.Beta))

    @classmethod
    def _rvslpmf_all(cls, numpy.ndarray[dtype=numpy.float64_t, ndim=2] Beta,
                          numpy.ndarray[dtype=numpy.float64_t, ndim=2] U=None,
                          numpy.ndarray[dtype=numpy.int8_t, ndim=2] Y=None):
        """
            All-purpose routine for sampling and point-wise evaluation.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        cdef Py_ssize_t d = Beta.shape[0]
        cdef Py_ssize_t k, i, size
        cdef double logprob
        cdef double x

        if U is not None:
            size = U.shape[0]
            Y = numpy.empty((size, d), dtype=numpy.int8)

        if Y is not None:
            size = Y.shape[0]

        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] L = numpy.zeros(size, dtype=numpy.float64)

        for k in xrange(size):

            for i in xrange(d):
                # Compute log conditional probability that Y(i) is one
                x = Beta[i, i]
                for j in xrange(i):
                    x += Beta[i, j] * Y[k, j]
                logcprob = -log(1 + exp(-x))

                # Generate the ith entry
                if U is not None: Y[k, i] = log(U[k, i]) < logcprob

                # Add to log conditional probability
                L[k] += logcprob
                if not Y[k, i]: L[k] -= x

        return numpy.array(Y, dtype=bool), L

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
        for i in xrange(d): Beta[i, i] = cls.Beta[i, i]
        cls.Beta = Beta
        return cls

    @classmethod
    def from_moments(cls, mean, corr, n=1e4, q=25.0, delta=None, verbose=False):
        """ 
            Constructs a logistic conditionals family from given mean and correlation.
            \param mean mean
            \param corr correlation
            \param n number of samples for Monte Carlo estimation
            \param q number of intermediate steps in Newton-Raphson procedure
            \param delta minimum absolute value of correlation coefficients
            \return logistic conditionals family
        """

        ## dimension of binary family
        cdef Py_ssize_t d = mean.shape[0]

        ## dimension of the current logistic regression
        cdef Py_ssize_t c

        ## dimension of the sparse logistic regression
        cdef Py_ssize_t s

        ## iterators
        cdef Py_ssize_t k, i, j

        ## minimum dimension for Monte Carlo estimates
        cdef Py_ssize_t min_c = int(numpy.log2(n))

        ## probability of binary vector
        cdef double prob

        ## floating point variable
        cdef double x, high, low

        ## parameter matrix holding regression coefficients
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] Beta = numpy.zeros((d, d), dtype=numpy.float64)

        ## parameter vector holding regression coefficients
        cdef numpy.ndarray[numpy.float64_t, ndim = 1] beta = numpy.empty(0, dtype=numpy.float64)

        ## f: mapping of the parameter vector onto the cross-moment vector
        cdef numpy.ndarray[numpy.float64_t, ndim = 1] f = numpy.empty(0, dtype=numpy.float64)

        ## Jacobian matrix of f
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] J = numpy.empty((0, 0), dtype=numpy.float64)

        ## probability mass for enumeration 
        cdef numpy.ndarray[numpy.float64_t, ndim = 1] pm = numpy.empty(0, dtype=numpy.float64)

        ## array holding uniform [0,1] random variables
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] U = numpy.empty((0, 0), dtype=numpy.float64)

        ## array holding random binary vectors
        cdef numpy.ndarray[numpy.int8_t, ndim = 2] Y = numpy.empty((0, 0), dtype=numpy.int8)

        ## index vector for sparse logistic regression
        cdef numpy.ndarray[Py_ssize_t, ndim = 1] S = numpy.empty(0, dtype=numpy.int)

        ## index vector for intermediate steps in Newton-Raphson procedure
        cdef numpy.ndarray[numpy.float64_t, ndim = 1] Q = numpy.linspace(1 / (float(q) - 1), 1.0, q - 1)

        ## cross-moment matrix
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] M = binary.base.corr2moments(mean, corr)

        ## cross-moment matrix for independent components
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] I = M.copy()

        ## parameter for convex combination between target and independent moment vectors 
        cdef double phi


        #------------------------------------------------------------------------------ 

        if delta is None:
            delta = 2.0 * scipy.linalg.norm(numpy.tril(corr, k= -1) + numpy.triu(corr, k=1)) / float((d - 1) * d)

        # compute cross-moment for independent case
        for i in xrange(d):
            for j in xrange(i):
                I[i, j] = M[i, i] * M[j, j]
                I[j, i] = I[i, j]

        # initialize for component of Beta
        Beta[0, 0] = logit(M[0, 0])

        # loop over dimensions
        for c in xrange(1, d):
            if verbose > 0: sys.stderr.write('\ndim: %d' % c)

            if c < min_c:
                Y = numpy.array(LogisticCondBinary.state_space(c), dtype=numpy.int8)
                pm = numpy.exp(LogisticCondBinary._rvslpmf_all(Beta=Beta[:c, :c], Y=Y)[1])
            else:
                Y = numpy.empty(shape=(n, c), dtype=numpy.int8)
                U = numpy.random.random(size=(n, c))

                # sample array of random binary vectors
                for k in xrange(n):

                    for i in xrange(c):

                        # compute the probability that Y(k,i) is one                    
                        x = Beta[i, i]
                        for j in xrange(i): x += Beta[i, j] * Y[k, j]

                        # generate the entry Y(k,i)
                        Y[k, i] = U[k, i] < 1.0 / (1.0 + exp(-x))

            # filter components with high association for sparse regression
            S = numpy.append((abs(corr[c, :c]) > delta).nonzero(), c)
            s = S.shape[0] - 1

            # initialize b with independent parameter
            beta = numpy.zeros(s + 1, dtype=numpy.float64)
            beta[s] = logit(M[c, c])
            Beta[c, S] = beta

            # set target moment vector and independent moment vector
            tM, tI = M[c, S], I[c, S]

            # Newton-Raphson iteration
            for phi in Q:

                for nr in xrange(LogisticCondBinary.MAX_ITERATIONS):

                    if verbose > 1: sys.stderr.write('phi: %.3f, nr: %d, beta: %s' % (phi, nr, repr(beta)))
                    beta_before = beta.copy()

                    # compute f and J 
                    f = numpy.zeros(s + 1, dtype=numpy.float64)
                    J = numpy.zeros((s + 1, s + 1), dtype=numpy.float64)

                    # loop over all binary vectors
                    for k in xrange(Y.shape[0]):

                        x = beta[s]
                        for i in xrange(s): x += beta[i] * Y[k, S[i]]

                        prob = 1.0 / (1.0 + exp(-x))
                        prob = min(max(prob, 1e-8), 1.0 - 1e-8)

                        x = prob / float(n)
                        if c < min_c: x *= n * pm[k]
                        for i in xrange(s):
                            f[i] = f[i] + x * Y[k, S[i]]
                        f[s] = f[s] + x

                        x = prob * (1 - prob) / float(n)
                        if c < min_c: x *= n * pm[k]
                        for i in xrange(s):
                            for j in xrange(s):
                                J[i, j] += x * Y[k, S[i]] * Y[k, S[j]]
                            J[s, i] += x * Y[k, S[i]]
                            J[i, s] += x * Y[k, S[i]]
                        J[s, s] += x

                    # subtract non-random parts
                    f -= (phi * tM + (1.0 - phi) * tI)

                    # Newton update
                    try:
                        beta = scipy.linalg.solve(J, numpy.dot(J, beta) - f, sym_pos=True)
                    except numpy.linalg.linalg.LinAlgError:
                        sys.stderr.write('numerical error. adding 1e-8 on main diagonal.')
                        beta = scipy.linalg.solve(J + numpy.eye(s + 1) * 1e-8, numpy.dot(J, beta) - f, sym_pos=False)

                    # check for absolute sums in beta
                    entry_sum = max(beta[beta > 0].sum(), -beta[beta < 0].sum())
                    if entry_sum > LogisticCondBinary.MAX_ENTRY_SUM * (0.25 * s + 1):
                        if verbose > 1: sys.stderr.write('stopped. beta exceeding %.1f\n' % entry_sum)
                        nr = None
                        break

                    # check for convergence
                    if numpy.allclose(beta, beta_before, rtol=0, atol=LogisticCondBinary.PRECISION):
                        if verbose > 1: sys.stderr.write('beta converged.\n')
                        Beta[c, S] = beta
                        break

                if nr is None or phi == 1.0:
                    if verbose: sys.stderr.write('phi: %.3f ' % phi)
                    break


        if verbose: sys.stderr.write('\nlogistic conditionals family successfully constructed from moments.\n\n')

        return cls(Beta)

    @classmethod
    def from_qu_exponential(cls, qu_exp):
        """ 
            Constructs a logistic conditionals family that approximates a
            quadratic exponential family.
            \param qu_exp quadratic exponential family
            \todo Instead of margining out in arbitrary order, we could use a greedy approach
            which picks the next dimension by minimizing the error made in the Taylor approximation. 
        """
        d = qu_exp.d
        Beta = numpy.zeros((d, d))
        Beta[0, 0] = logit(qu_exp.p_0)

        A = numpy.copy(qu_exp.A)
        for i in xrange(d - 1, 0, -1):
            Beta[i, 0:i] = A[i, :i] * 2.0
            Beta[i, i] = A[i, i]
            A = binary.qu_exponential.calc_marginal(A)

        return cls(Beta)

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
        newBeta = calc_Beta(sample=sample, Init=self.Beta,
                            job_server=job_server, eps=eps, delta=delta, verbose=verbose, pywrapper=self.py_wrapper)

        # Set convex combination of old and new parameter.
        self.Beta = (1 - lag) * newBeta + lag * self.Beta
        self.p = logistic(self.Beta.sum(axis=1))

    @classmethod
    def test_properties(cls, d, n=1e4, phi=0.8, ncpus=1):
        """
            Tests functionality of the quadratic linear family class.
            \param d dimension
            \param n number of samples
            \param phi dependency level in [0,1]
            \param ncpus number of cpus 
        """

        mean, corr = binary.base.moments2corr(binary.base.random_moments(d, phi=phi))
        print 'given marginals '.ljust(100, '*')
        binary.base.print_moments(mean, corr)

        generator = LogisticCondBinary.from_moments(mean, corr)
        print generator.name + ':'
        print generator

        #print 'exact '.ljust(100, '*')
        #binary.base.print_moments(generator.exact_marginals(ncpus))

        print ('simulation (n = %d) ' % n).ljust(100, '*')
        binary.base.print_moments(generator.rvs_marginals(n, ncpus))


def calc_Beta(sample, eps=0.02, delta=0.05, Init=None, job_server=None, verbose=True, pywrapper=None):
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
    logit_p = logit(p)

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
                         func=pywrapper.calc_log_regr,
                         args=(X[:, i], X[:, covariates + [d]], XW[:, covariates + [d]], Init[i, covariates + [i]], w, False),
                         modules=('numpy', 'scipy.linalg', 'binary.logistic_cond')))
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

        if numpy.allclose(last_beta, beta, rtol=0, atol=LogisticCondBinary.PRECISION):
            if verbose: print 'no change in beta\n'
            break

        if verbose: print '%i) log-likelihood %.2f' % (i + 1, llh)
        if abs(last_llh - llh) < LogisticCondBinary.PRECISION:
            if verbose: print 'no change in likelihood\n'
            break
    return beta, i + 1

def logistic(x):
    """ Logistic function 1/(1+exp(x)) \return logistic function """
    return 1.0 / (1.0 + numpy.exp(-x))

def logit(p):
    """ Logit function exp(1/(1-p)) \return logit function """
    return numpy.log(p / (1.0 - p))
