#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with logistic conditionals. \namespace binary.conditionals_logistic"""

import numpy
cimport numpy

cdef extern from "math.h":
    double exp(double)
    double log(double)

import scipy.linalg
import sys
import time
import utils

import binary.conditionals as conditionals
import binary.base as base
import binary.quadratic_exponential as exponential
import binary.wrapper as wrapper

class LogisticCondBinary(conditionals.ConditionalsBinary):
    """ Binary parametric family with logistic conditionals. """

    def __init__(self, A, name='logistic conditionals family', long_name=__doc__):
        """ 
            Constructor.
            \param A Lower triangular matrix holding regression coefficients
            \param name name
            \param long_name long name
        """

        # call super constructor
        super(LogisticCondBinary, self).__init__(A=A, name=name, long_name=long_name)

        # add modules
        self.py_wrapper = wrapper.conditionals_logistic()
        self.pp_modules = ('numpy', 'scipy.linalg', 'binary.conditionals_logistic',)

    @classmethod
    def from_exponential(cls, myexp):
        """ 
            Constructs a logistic conditionals family that approximates a
            quadratic exponential family.
            \param myexp quadratic exponential family
            \todo Instead of margining out in arbitrary order, we could use a greedy approach
            which picks the next dimension by minimizing the error made in the Taylor approximation. 
        """
        d = myexp.d
        A = numpy.zeros((d, d))
        A[0, 0] = cls.ilink(myexp.p_0)

        A = numpy.copy(myexp.A)
        for i in xrange(d - 1, 0, -1):
            A[i, 0:i] = A[i, :i] * 2.0
            A[i, i] = A[i, i]
            A = exponential.calc_marginal(A)

        return cls(A)

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
        return cls(LogisticCondBinary.calc_A(sample=sample, Init=Init,
                                             job_server=job_server, eps=eps, delta=delta,
                                             verbose=verbose))

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
        newA = LogisticCondBinary.calc_A(sample=sample, Init=self.A,
                                         job_server=job_server, eps=eps, delta=delta,
                                         verbose=verbose, pywrapper=self.py_wrapper)

        # Set convex combination of old and new parameter.
        self.A = (1 - lag) * newA + lag * self.A
        self.p = LogisticCondBinary.link(self.A.sum(axis=1))

    @classmethod
    def calc_A(cls, sample, eps=0.02, delta=0.05, Init=None, job_server=None, verbose=True, pywrapper=None):
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
    
        # Compute logits
        logit_p = cls.ilink(p)
    
        # check whether data is sufficient to fit logistic model
        if n == 1: return numpy.diag(cls.ilink_p)
    
        # Find strong associations.
        L = abs(utils.data.calc_cor(X[:, 0:d], w=w)) > delta
    
        # Remove components with extreme marginals.
        for i, m in enumerate(p):
            if m < eps or m > 1.0 - eps:
                L[i, :] = numpy.zeros(d, dtype=bool)
    
        # Initialize A with logits on diagonal.
        A = numpy.zeros((d, d))
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
                             func=LogisticCondBinary.calc_log_regr,
                             args=(X[:, i], X[:, covariates + [d]], XW[:, covariates + [d]], Init[i, covariates + [i]], w, False),
                             modules=('numpy', 'scipy.linalg', 'binary.logistic')))
                    ])
                    # once jobs are assigned to all cpus let the job server
                    # wait in order to prevent memory errors on large problems
                    ncpus -= 1
                    if ncpus <= 0:
                        job_server.wait()
                        ncpus = job_server.get_ncpus()
                else:
                    jobs.append([i, covariates + [i], (LogisticCondBinary.calc_log_regr(
                                  X[:, i], X[:, covariates + [d]], XW[:, covariates + [d]], Init[i, covariates + [i]], w, False))
                    ])
            else:
                A[i, i] = logit_p[i]
    
        # wait and retrieve results
        if not job_server is None: job_server.wait()
    
        # write results to A matrix
        for i, covariates, job in jobs:
            if isinstance(job, tuple): a, iterations = job
            else: a, iterations = job()
    
            if verbose:
                stats['iterations'] = iterations
                stats['regressions'] += 1.0
                stats['product'] -= 1
                stats['logistic'] += 1
    
            if not a is None:
                A[i, covariates] = a
            else:
                if verbose: stats['failures'] += 1
                A[i, i] = cls.ilink_p[i]
    
        if verbose:
            stats.update({'time':time.time() - t})
            if stats['regressions'] > 0: stats['iterations'] /= stats['regressions']
            print 'Logistic model: (p %(product)i, l %(logistic)i), loops %(iterations).3f, failures %(failures)i, time %(time).3f\n' % stats
    
        return A
    
    @classmethod
    def calc_log_regr(cls, y, X, XW, init, w=None, verbose=False):
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
        a = init
        if w is None: w = numpy.ones(n) / float(n)
        v = numpy.empty(n)
        P = numpy.empty(n)
        llh = -numpy.inf
        _lambda = 1e-8
    
        for i in xrange(50):
    
            # Save last iterations values.
            last_llh = llh
            last_a = a.copy()
    
            # Compute Fisher information at a
            Xa = numpy.dot(X, a)
            Xa = numpy.minimum(numpy.maximum(Xa, -500), 500)
    
            p = numpy.power(1 + numpy.exp(-Xa), -1)
            P = p * (1 - p)
            XWPX = numpy.dot(XW.T, P[:, numpy.newaxis] * X) + _lambda * numpy.eye(d)
            v = P * Xa + y - p
    
            # Solve Newton-Raphson equation.
            try:
                a = scipy.linalg.solve(XWPX, numpy.dot(XW.T, v), sym_pos=True)
            except:
                if verbose: print '> likelihood not unimodal'
                a = scipy.linalg.solve(XWPX, numpy.dot(XW.T, v), sym_pos=False)
    
            # Compute the log-likelihood.
            llh = -0.5 * _lambda * numpy.dot(a, a) + (w * (y * Xa + numpy.log(1 + numpy.exp(Xa)))).sum()
            if abs(a).max() > 1e4:
                if verbose: print 'convergence failure\n'
                return None, i
    
            if numpy.allclose(last_a, a, rtol=0, atol=LogisticCondBinary.PRECISION):
                if verbose: print 'no change in a\n'
                break
    
            if verbose: print '%i) log-likelihood %.2f' % (i + 1, llh)
            if abs(last_llh - llh) < LogisticCondBinary.PRECISION:
                if verbose: print 'no change in likelihood\n'
                break
        return a, i + 1

    @classmethod
    def link(cls, x):
        """ Logistic function 1/(1+exp(x)) \return logistic function """
        return 1.0 / (1.0 + numpy.exp(-x))

    @classmethod
    def dlink(cls, x):
        """ Derivative of logistic function 1/(1+exp(x)) \return derivative of logistic function """
        p = LogisticCondBinary.link(x)
        return p * (1 - p)

    @classmethod
    def ilink(cls, p):
        """ Inverse of logistic function exp(1/(1-p)) \return logit function """
        return numpy.log(p / (1.0 - p))