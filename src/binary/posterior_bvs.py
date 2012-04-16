#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Hierarchical Bayesian posterior.
    @namespace binary.posterior_bvs
"""

import numpy
import scipy.linalg
import binary.posterior as posterior
import wrapper

PRIOR_ZELLNER = 1
PRIOR_INDEPENDENT = 2

class PosteriorBVS(posterior.Posterior):
    """ Hierarchical Bayesian posterior. """

    name = 'hierarchical Bayesian posterior.'

    def __init__(self, y, Z, config, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param y explained variable
            \param Z covariates to perform selection on
            \param config parameter dictionary
        """

        super(PosteriorBVS, self).__init__(y, Z, config, name=name, long_name=long_name)

        # add modules
        self.py_wrapper = wrapper.posterior_bvs()
        self.pp_modules += ('binary.posterior_bvs',)

        # prior covariance on beta
        self.tau = config['prior/var_dispersion']
        if isinstance(self.tau, str):
            if self.tau == 'n': self.tau = float(self.n)

        if config['prior/cov_matrix_hp'] == 'zellner':
            self.PRIOR = PRIOR_ZELLNER
            self.W = (1.0 + 1.0 / self.tau) * numpy.dot(Z.T, Z) + 1e-10 * numpy.eye(Z.shape[1])
        if config['prior/cov_matrix_hp'] == 'independent':
            self.PRIOR = PRIOR_INDEPENDENT
            self.W = numpy.dot(Z.T, Z) + (1.0 / self.tau) * numpy.eye(Z.shape[1])

        # prior on sigma
        a = config['prior/var_hp_a']
        b = config['prior/var_hp_b']

        # prior on Y
        p = config['prior/model_inclprob_hp']

        # constants
        self.A_TIMES_B_PLUS_TSS = a * b + self.tss
        self.NEG_HALF_N_MINUS_ONE_PLUS_A = -0.5 * (self.n - 1.0 + a)
        self.NEG_HALF_LOG_ONE_PLUS_TAU = -0.5 * numpy.log(1.0 + self.tau)
        self.NEG_HALF_LOG_TAU = -0.5 * numpy.log(self.tau)
        self.LOGIT_P = numpy.log(p / (1.0 - p))

    def univariate_bayes(self):
        """ 
            Setup univariate Hierarchical Bayesian posterior.
      
            \param parameter dictionary
        """

        d = self.Z.shape[1]
        T = numpy.empty(d)

        param = self.param.copy()
        param.update({'d':1})

        Y = numpy.array([True])[:, numpy.newaxis]
        log_prob_H0 = float(self._lpmf(self, Y - 1, param))

        for i in xrange(d):
            param.update({'Zty':numpy.dot(self.Z[:, i].T, self.y)[numpy.newaxis, numpy.newaxis],
                           'W': numpy.dot(self.Z[:, i].T, self.Z[:, i])[numpy.newaxis, numpy.newaxis] + 1e-10})
            log_prob_H1 = float(self._lpmf(self, Y - 1, param))
            m = max(log_prob_H0, log_prob_H1)
            prob_H0 = numpy.exp(log_prob_H0 - m)
            prob_H1 = numpy.exp(log_prob_H1 - m)
            T[i] = prob_H1 / (prob_H0 + prob_H1)

        return T

    @classmethod
    def score(cls, Ystar, config, size):
        btb = 0.0
        if size > 0:
            C = scipy.linalg.cholesky(config.W[Ystar, :][:, Ystar])
            b = scipy.linalg.solve(C.T, config.Zty[Ystar, :])
            btb = numpy.dot(b.T, b)

        L = config.NEG_HALF_N_MINUS_ONE_PLUS_A * numpy.log(config.A_TIMES_B_PLUS_TSS - btb)
        
        # size penalty
        L += size * config.LOGIT_P

        # difference between Zellner's and independent prior
        if config.PRIOR == PRIOR_ZELLNER:
            L += size * config.NEG_HALF_LOG_ONE_PLUS_TAU
        if config.PRIOR == PRIOR_INDEPENDENT:
            L += size * config.NEG_HALF_LOG_TAU
            if size > 0: L -= numpy.log(C.diagonal()).sum()

        return L
