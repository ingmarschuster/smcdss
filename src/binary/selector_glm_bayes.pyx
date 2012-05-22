#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 
    Bayesian variable selection for generalized linear models.
    @namespace binary.selector_glm_bayes
"""
from binary import selector_glm

cimport numpy
import binary.selector_glm as glm
import binary.wrapper as wrapper
import numpy
import scipy.linalg
import sys

cdef extern from "math.h":
    float exp(float)
    float log(float)
    float sqrt(float)

cdef enum:
    FULL_IS = 1
    LAPLACE_PLUS_IS = 2
    LAPLACE = 3
    NORMAL = 4
    STUDENT = 5

cdef float PI = 3.141592653589793115997963468544185161590576171875
cdef float HALF_LOG_2_PI = 0.918938533204672669540968854562379419803619384765625
cdef float NU = 5.0

class SelectorGmlBayes(glm.SelectorGlm):
    """ Criteria based on maximum likelihood for generalized linear models."""

    name = 'hierarchical Bayesian posterior glm'

    def __init__(self, y, Z, config, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param Y explained variable
            \param Z covariates to perform selection on
            \param config dictionary
        """

        super(SelectorGmlBayes, self).__init__(y, Z, config, name=name, long_name=long_name)

        # add modules
        self.py_wrapper = wrapper.selector_glm_bayes()
        self.pp_modules += ('binary.selector_glm_bayes',)

        # normalize
        self.Z = numpy.column_stack((numpy.ones(Z.shape[0]), Z))
        self.y = y
        self.n = Z.shape[0]

        # prior on sigma
        if isinstance(config['prior/var_dispersion'], str):
            self.tau = float(self.n)
        else:
            self.tau = float(config['prior/var_dispersion'])
        self.HALF_LOG_2_PI_TAU = 0.5 * log(2 * PI * self.tau)

        # ess for importance sampling
        if config['prior/is_ess'] in ['', None]:
            self.n_samples = Z.shape[1]
        else:
            self.n_samples = int(config['prior/is_ess'])

        # proposal for importance sampling
        self.proposal = {
            'normal': NORMAL,
            'student': STUDENT
        }[config['prior/is_proposal'].lower()]

        # approximation
        self.criterion = {
            'laplace': NORMAL,
            'laplace+is': LAPLACE_PLUS_IS,
            'full is': FULL_IS
        }[config['prior/criterion'].lower()]

    def score(self, numpy.ndarray[dtype=numpy.float64_t, ndim=1] y,
                    numpy.ndarray[dtype=numpy.float64_t, ndim=2] Z,
                    numpy.ndarray[dtype=numpy.int16_t, ndim=1] index):
        """ 
            Compute the score of the model.
            \param y explained variable
            \param Z covariates
            \param index indicators of submodel
        """

        if self.criterion == FULL_IS:
            return self.is_estimate(y, Z, index)

        # compute MLE
        beta_mle, F_mle = self.compute_mle(y, Z, index)

        # compute determinant
        try:
            half_log_det_F_mle = 0.5 * numpy.linalg.slogdet(F_mle)[1]
        except AttributeError:
            half_log_det_F_mle = numpy.log(scipy.linalg.cholesky(F_mle).diagonal()).sum()

        return self.log_llh(y, Z, beta_mle, index) + index.shape[0] * HALF_LOG_2_PI - half_log_det_F_mle

    def is_estimate(self, numpy.ndarray[dtype=numpy.float64_t, ndim=1] y,
                          numpy.ndarray[dtype=numpy.float64_t, ndim=2] Z,
                          numpy.ndarray[dtype=numpy.int16_t, ndim=1] index):
        """ 
            Compute an importance sampling approximation of the normalizing
            constant of the Bayesian posterior for the given submodel.
            \param y explained variable
            \param Z covariates
            \param index indicators of submodel
        """

        cdef Py_ssize_t d, i, j
        cdef float log_proposal, log_target, log_det_F, x_F_x, const, loop, w, m, s1, s2
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] F_mle
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] C_inv
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] x
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] g

        d = index.shape[0]

        beta_mle, F_mle = self.compute_mle(y, Z, index)

        # Cholesky decomposition
        C_inv = scipy.linalg.cholesky(scipy.linalg.inv(F_mle), lower=True)

        # log determinant of Fisher matrix
        log_det_F = -2.0 * numpy.log(C_inv.diagonal()).sum()

        if self.proposal == NORMAL:
            const = -d * HALF_LOG_2_PI + 0.5 * log_det_F
        if self.proposal == STUDENT:
            const = scipy.special.gammaln((NU + d) / 2.0) - 0.5 * d * log(NU * PI) + 0.5 * log_det_F

        loop = 0.0
        s1 = 0.0
        s2 = 0.0
        m = -numpy.inf

        while ess < self.n_samples and loop < self.n_samples * 20:

            # sample normal vector
            g = numpy.random.normal(size=d)
            x = numpy.zeros(d)
            for i from 0 <= i < d:
                for j from 0 <= j <= i:
                    x[i] += g[j] * C_inv[i, j]

            # sample student vector
            if not NORMAL:
                x *= sqrt(NU / numpy.random.chisquare(df=NU))

            # compute bilinear form
            x_F_x = 0.0
            for i from 0 <= i < d:
                x_F_x += F_mle[i, i] * x[i] * x[i]
                for j from 0 <= j < i:
                    x_F_x += 2.0 * F_mle[i, j] * x[i] * x[j]

            # compute mass of proposal
            log_proposal = const

            if NORMAL:
                log_proposal += -0.5 * x_F_x
            else:
                log_proposal += -0.5 * (NU + d) * log(1.0 + x_F_x / NU)

            # compute mass of posterior
            log_target = self.log_llh(y, Z, beta=beta_mle + x, index=index)

            # store log-weight
            w = log_target - log_proposal

            if w > m:
                xi = exp(m - w)
                s1 *= xi
                s2 *= xi * xi
                m = w
                w = 1.0
            else:
                w = exp(w - m)

            loop += 1.0
            s1 += w
            s2 += w * w
            ess = (s1 * s1) / s2

        #print loop, ess, self.log_llh(y, Z, beta=beta_mle, index=index), log_target, index

        return log(s1 / loop) + m
