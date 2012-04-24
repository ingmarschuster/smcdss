#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 
    Variable selector for generalized linear models.
    @namespace binary.selector_glm
"""

cimport numpy
import binary.base as base
import binary.wrapper as wrapper
import numpy
import os
import scipy.linalg
import scipy.special
import sys

cdef extern from "math.h":
    float exp(float)
    float log(float)

class SelectorGlm(base.BaseBinary):
    """ Variable selector for generalized linear models."""

    name = 'selector glm'

    def __init__(self, y, Z, config, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param Y explained variable
            \param Z covariates to perform selection on
            \param config dictionary
        """

        super(SelectorGlm, self).__init__(d=Z.shape[1], name=name, long_name=long_name)

        # add modules
        self.pp_modules += ('binary.selector_glm',)

        # add constant column
        self.Z = numpy.column_stack((numpy.ones(Z.shape[0]), Z))
        self.n = Z.shape[0]
        self.y = y

        # prior on marginal inclusion probability
        p = config['prior/model_inclprob']
        if p is None: self.LOGIT_P = 0.0
        else: self.LOGIT_P = numpy.log(p / (1.0 - p))

        # empty model
        self.ILINK = ilink(self.y.sum() / float(self.n))

    @classmethod
    def _lpmf(cls, numpy.ndarray[dtype=numpy.int8_t, ndim=2] Y, config):
        """ 
            Log-posterior probability mass function.
            
            \param Y binary vector
            \param param parameters
            \return log-probabilities
        """

        cdef Py_ssize_t d, i, j

        # array to store results
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] L = numpy.empty(Y.shape[0], dtype=float)
        # model index
        cdef numpy.ndarray[dtype = numpy.int16_t, ndim = 1] index
        # c version of observations
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] y = config.y
        # c version of predictors
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] Z = config.Z

        # loop over all models
        for k in xrange(Y.shape[0]):

            # determine indices
            d, j = 1 + Y[k].sum(), 1
            index = numpy.zeros(d, dtype=numpy.int16)
            for i in xrange(config.d):
                if Y[k, i]:
                    index[j] = i + 1
                    j += 1

            L[k] = d * config.LOGIT_P
            L[k] += config.score(y, Z, index)

        return L

    def bvs(self, numpy.ndarray[dtype=numpy.float64_t, ndim=1] y,
                        numpy.ndarray[dtype=numpy.float64_t, ndim=2] Z,
                        numpy.ndarray[dtype=numpy.int16_t, ndim=1] index):

        cdef Py_ssize_t d = index.shape[0]
        cdef Py_ssize_t i, j
        cdef float log_proposal, log_target

        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] w = numpy.empty(self.n_samples)
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] F_mle

        beta_mle, F_mle = self.compute_mle(y, Z, index)
        C = scipy.linalg.cholesky(scipy.linalg.inv(F_mle), lower=True)

        for k in xrange(self.n_samples):

            # compute Gaussian proposal
            x = numpy.dot(C, numpy.random.normal(size=d))
            log_proposal = 0.0
            for i in xrange(d):
                log_proposal -= 0.5 * F_mle[i, i] * x[i] * x[i]
                for j in xrange(i):
                    log_proposal -= F_mle[i, j] * x[i] * x[j]

            # compute log-likelihood
            log_target = self.log_llh(y, Z, beta=beta_mle + x, index=index)

            # store log-weight
            w[k] = log_target - log_proposal

        # compute average
        m = w.max()
        return log(numpy.exp(w - m).sum()) + m


    def compute_mle(self, numpy.ndarray[dtype=numpy.float64_t, ndim=1] y,
                            numpy.ndarray[dtype=numpy.float64_t, ndim=2] Z,
                            numpy.ndarray[dtype=numpy.int16_t, ndim=1] index):
        """
            Compute maximum likelihood estimator
            \return maximum log-likelihood, maximum likelihood estimator, Fisher matrix
        """

        cdef Py_ssize_t d = index.shape[0]

        # coefficients
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] beta = numpy.zeros(d)
        # score 
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] s = numpy.empty(d)
        # Fisher matrix
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] F = numpy.empty((d, d))

        # initialize beta
        beta[0] = self.ILINK

        # Newton-Raphson iterations
        while True:
            beta_before = beta.copy()

            # score and Fisher matrix
            s, F = self.d_log_llh(y, Z, beta, index)

            # Newton-Raphson update
            beta += scipy.linalg.solve(F, s, sym_pos=True)
            if numpy.allclose(beta_before, beta, rtol=0, atol=self.PRECISION):
                break

        return beta, F


    def log_llh(self, numpy.ndarray[dtype=numpy.float64_t, ndim=1] y,
                            numpy.ndarray[dtype=numpy.float64_t, ndim=2] Z,
                            numpy.ndarray[dtype=numpy.float64_t, ndim=1] beta,
                            numpy.ndarray[dtype=numpy.int16_t, ndim=1] index):
        """
            Compute the log-likelihood.
            \param y observations
            \param Z predictors
            \param beta coefficients
            \param index model
        """
        cdef float zbeta, mu, v, s
        cdef Py_ssize_t i, k
        cdef Py_ssize_t d = index.shape[0]

        v = 0.0
        for k in xrange(self.n):

            zbeta = 0.0
            for i in xrange(d):
                zbeta += Z[k, index[i]] * beta[i]
            mu = c_link(zbeta)

            if y[k]:
                v += log(mu)
            else:
                v += log(1 - mu)

        # prior
        if self.prior and d > 1:
            s = (d + self.a - 1) / 2.0
            v += scipy.special.gammaln(s)
            v -= s * log(numpy.pi * (2 * self.b + numpy.dot(beta, beta)))

        return v

    def d_log_llh(self, numpy.ndarray[dtype=numpy.float64_t, ndim=1] y,
                                        numpy.ndarray[dtype=numpy.float64_t, ndim=2] Z,
                                        numpy.ndarray[dtype=numpy.float64_t, ndim=1] beta,
                                        numpy.ndarray[dtype=numpy.int16_t, ndim=1] index):
        """
            Compute the derivatives of the log-likelihood function.
            \param y observations
            \param Z predictors
            \param beta coefficients
            \param index model
            \return score vector, Fisher matrix
        """

        cdef Py_ssize_t i, j, k
        cdef Py_ssize_t d = index.shape[0]
        cdef float zbeta, mu, v1, v2

        # score vector
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] s = numpy.zeros(d)
        # Fisher matrix
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] F = numpy.zeros((d, d))

        for k in xrange(self.n):

            # compute linear combination
            zbeta = 0.0
            for i in xrange(d):
                zbeta += Z[k, index[i]] * beta[i]

            # compute link
            mu = c_link(zbeta)

            # compute summands
            if self.y[k]:
                v1 = c_1der_link(zbeta) / mu
                v2 = c_2der_link(zbeta) / mu - v1 * v1
            else:
                v1 = -c_1der_link(zbeta) / (1 - mu)
                v2 = -c_2der_link(zbeta) / (1.0 - mu) - v1 * v1

            # add summands
            for i in xrange(d):
                s[i] += Z[k, index[i]] * v1
                F[i, i] -= Z[k, index[i]] * Z[k, index[i]] * v2
                for j in xrange(i):
                    F[i, j] -= Z[k, index[i]] * Z[k, index[j]] * v2

        # Use symmetry of Fisher matrix
        for i in xrange(d):
            for j in xrange(i):
                F[j, i] = F[i, j]

        # prior
        if self.prior and d + self.a > 1:
            beta2 = numpy.dot(beta, beta) + 2 * self.b
            p1 = (d + self.a - 1) / beta2
            p2 = 2 * p1 / beta2
            for i in xrange(d):
                # adjust score
                s[i] -= p1 * beta[i]

                # adjust Fisher matrix
                F[i, i] += p1 * beta[i]
                for j in xrange(d):
                    F[i, j] -= p2 * beta[i] * beta[j]

        return s, F

def link(x):
    """ Link. """
    return c_link(x)

def ilink(p):
    """ Inverse """
    return log(p / (1.0 - p))

cdef float c_link(float x):
    """ Link """
    return 1.0 / (1.0 + exp(-x))

cdef float c_1der_link(float x):
    """ 1 Derivative. """
    v = c_link(x)
    return v * (1 - v)

cdef float c_2der_link(float x):
    """ 2 Derivative. """
    v = c_link(x)
    return v * (1 - v) * (1 - 2 * v)
