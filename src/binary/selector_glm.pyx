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
from scipy.special._cephes import ndtri
import sys

cdef extern from "math.h":
    float exp(float)
    float log(float)
    float erf(float)

cdef float SQRT_2_PI = 2.506628274631000241612355239340104162693023681640625
cdef float SQRT_2 = 1.4142135623730951454746218587388284504413604736328125
cdef int MAXIMUM_ITERATIONS = 20

cdef int LOGISTIC = 1
cdef int PROBIT = 2

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
        self.pp_modules += ('binary.selector_glm', 'scipy.linalg',)

        # add constant column
        self.Z = numpy.column_stack((numpy.ones(Z.shape[0]), Z))
        self.n = Z.shape[0]
        self.y = y

        if config['prior/model'].lower() == 'logistic':
            self.link = LOGISTIC
        if config['prior/model'].lower() == 'probit':
            self.link = PROBIT

        # prior on marginal inclusion probability
        p = config['prior/model_inclprob']
        self.LOGIT_P = log(p / (1.0 - p))

        # empty model
        self.tau = -1
        if self.link == LOGISTIC:
            self.const = ilogistic(self.y.sum() / float(self.n))
        if self.link == PROBIT:
            self.const = iprobit(self.y.sum() / float(self.n))

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
        for k from 0 <= k < Y.shape[0]:

            # determine indices
            d, j = 1 + Y[k].sum(), 1
            index = numpy.zeros(d, dtype=numpy.int16)
            for i from 0 <= i < config.d:
                if Y[k, i]:
                    index[j] = i + 1
                    j += 1
            L[k] = config.score(y, Z, index) + d * config.LOGIT_P

        return L

    def compute_mle(self, numpy.ndarray[dtype=numpy.float64_t, ndim=1] y,
                          numpy.ndarray[dtype=numpy.float64_t, ndim=2] Z,
                          numpy.ndarray[dtype=numpy.int16_t, ndim=1] index):
        """
            Compute maximum likelihood estimator
            \return maximum log-likelihood, maximum likelihood estimator, Fisher matrix
        """

        cdef Py_ssize_t d
        cdef Py_ssize_t k
        cdef float tau

        d = index.shape[0]

        # coefficients
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] beta = numpy.zeros(d)
        # score 
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] s = numpy.empty(d)
        # Fisher matrix
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] F = numpy.empty((d, d))

        # initialize beta
        beta[0] = self.const

        # Newton-Raphson iterations
        for k from 0 <= k < MAXIMUM_ITERATIONS:
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
        cdef Py_ssize_t i, k, d, n
        cdef int link

        d = index.shape[0]
        n = Z.shape[0]
        link = self.link
        v = 0.0

        for k from 0 <= k < n:
            zbeta = 0.0
            for i from 0 <= i < d:
                zbeta += Z[k, index[i]] * beta[i]

            if link == LOGISTIC: mu = logistic(zbeta)
            if link == PROBIT: mu = probit(zbeta)

            if y[k]:
                v += log(mu)
            else:
                v += log(1.0 - mu)

        # normal prior
        if self.tau > -1:
            v -= self.HALF_LOG_2_PI_TAU * d
            v -= 0.5 * c_inner(beta) / self.tau

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
                \param tau dispersion parameter
                \return score vector, Fisher matrix
            """

            cdef Py_ssize_t i, j, k, n, d
            cdef int link
            cdef float zbeta, mu, c1der_link, c2der_link, s1, s2, s_add, f_add

            n = Z.shape[0]
            d = index.shape[0]
            link = self.link

            # score vector
            cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] s = numpy.zeros(d)
            # Fisher matrix
            cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] F = numpy.zeros((d, d))

            for k from 0 <= k < n:

                # compute linear combination
                zbeta = 0.0
                for i from 0 <= i < d:
                    zbeta += Z[k, index[i]] * beta[i]

                # compute summands
                if link == LOGISTIC:
                    mu = logistic(zbeta)
                    c1der_link = logistic_1der(zbeta)
                    c2der_link = logistic_2der(zbeta)
                if link == PROBIT:
                    mu = probit(zbeta)
                    c1der_link = probit_1der(zbeta)
                    c2der_link = probit_2der(zbeta)
                s1 = c1der_link / mu
                s2 = c1der_link / (1.0 - mu)

                if y[k]:
                    s_add = s1
                    f_add = c2der_link / mu - s1 * s1
                else:
                    s_add = -s2
                    f_add = -c2der_link / (1.0 - mu) - s2 * s2

                '''
                s_add = y[k] * s1 - (1.0 - y[k]) * s2
                f_add = (
                        y[k] * (c2der_link / mu - s1 * s1) -
                        (1.0 - y[k]) * (c2der_link / (1.0 - mu) + s2 * s2)
                        )
                '''

                # add summands
                for i from 0 <= i < d:
                    s[i] += Z[k, index[i]] * s_add
                    F[i, i] -= Z[k, index[i]] * Z[k, index[i]] * f_add
                    for j from 0 <= j < i:
                        F[i, j] -= Z[k, index[i]] * Z[k, index[j]] * f_add

            # use symmetry of Fisher matrix
            for i from 0 <= i < d:
                for j from 0 <= j < i:
                    F[j, i] = F[i, j]

            # normal prior
            if self.tau > -1:
                for i from 0 <= i < d:
                    # adjust score
                    s[i] -= beta[i] / self.tau
                    # adjust Fisher matrix
                    F[i, i] += 1.0 / self.tau
            
            return s, F

def w_probit(x):
    return probit(x)
def w_logistic(x):
    return logistic(x)

cdef float iprobit(float p):
    """ Inverse """
    return ndtri(p)

cdef float probit(float x):
    """ Link """
    cdef float mu
    mu = 0.5 + 0.5 * erf(x / SQRT_2)
    if mu < 0.0000001: mu = 0.0000001
    if mu > 0.9999999: mu = 0.9999999
    return mu

cdef float probit_1der(float x):
    """ 1 Derivative. """
    return exp(-0.5 * x * x) / SQRT_2_PI

cdef float probit_2der(float x):
    """ 2 Derivative. """
    return -x * exp(-0.5 * x * x) / SQRT_2_PI


cdef float ilogistic(float p):
    """ Inverse """
    return log(p / (1.0 - p))

cdef float logistic(float x):
    """ Link """
    cdef float mu
    mu = 1.0 / (1.0 + exp(-x))
    if mu < 0.0000001: mu = 0.0000001
    if mu > 0.9999999: mu = 0.9999999
    return mu

cdef float logistic_1der(float x):
    """ 1 Derivative. """
    cdef float v
    v = logistic(x)
    return v * (1 - v)

cdef float logistic_2der(float x):
    """ 2 Derivative. """
    cdef float v
    v = logistic(x)
    return v * (1 - v) * (1 - 2 * v)

cdef float c_inner(numpy.ndarray[dtype=numpy.float64_t, ndim=1] beta):
    """ Link """
    cdef float v
    cdef long i
    for i from 0 <= i < beta.shape[0]:
        v += beta[i] * beta[i]
    return v
