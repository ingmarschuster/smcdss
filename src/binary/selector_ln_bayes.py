#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Hierarchical Bayesian posterior for linear normal models.
    @namespace binary.selector_ln_bayes
"""

import numpy
import scipy.linalg
import binary.selector_ln as ln
import wrapper

PRIOR_ZELLNER = 1
PRIOR_ZELLNER_INVGAMMA = 2
PRIOR_INDEPENDENT = 3

class SelectorLnBayes(ln.SelectorLn):
    """ Hierarchical Bayesian posterior. """

    name = 'hierarchical Bayesian posterior ln'

    def __init__(self, y, Z, config, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param y explained variable
            \param Z covariates to perform selection on
            \param config parameter dictionary
        """

        super(SelectorLnBayes, self).__init__(y, Z, config, name=name, long_name=long_name)

        # add modules
        self.py_wrapper = wrapper.selector_ln_bayes()
        self.pp_modules += ('binary.selector_ln_bayes',)

        # prior covariance on beta
        self.LOCAL_EMPIRICAL_BAYES = False
        if isinstance(config['prior/var_dispersion'], str):
            self.tau = float(self.n)
            if config['prior/var_dispersion'].lower() == 'leb': self.LOCAL_EMPIRICAL_BAYES = True
        else:
            self.tau = float(config['prior/var_dispersion'])

        if config['prior/cov_matrix_hp'].lower() == 'zellner':
            self.PRIOR = PRIOR_ZELLNER
        if config['prior/cov_matrix_hp'].lower() == 'zellner+invgamma':
            self.PRIOR = PRIOR_ZELLNER_INVGAMMA
        if config['prior/cov_matrix_hp'].lower() == 'independent':
            self.PRIOR = PRIOR_INDEPENDENT

        # prior covariance on beta
        self.W = numpy.dot(Z.T, Z) + 1e-10 * numpy.eye(Z.shape[1])
        if self.PRIOR == PRIOR_INDEPENDENT:
            self.W += (1.0 / self.tau) * numpy.eye(Z.shape[1])

        # prior on sigma
        self.a = config['prior/var_hp_a']
        self.b = config['prior/var_hp_b']

        # constants
        self.A_TIMES_B_PLUS_TSS = self.a * self.b + self.tss
        self.A_TIMES_B_PLUS_SCALED_TSS = self.a * self.b + (1.0 + 1.0 / self.tau) * self.tss
        self.NEG_HALF_N_MINUS_ONE_PLUS_A = -0.5 * (self.n - 1.0 + self.a)
        self.NEG_HALF_LOG_ONE_PLUS_TAU = -0.5 * numpy.log(1.0 + self.tau)
        self.NEG_HALF_LOG_TAU = -0.5 * numpy.log(self.tau)

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
    def zellner_invgamma_laplace(cls, n, size, R2):
        '''
            Computes the Laplace approximation for Zellner's prior with an
            inverse gamma IG(1/2,n/2) prior on the dispersion parameter tau.
            \param n number of observations
            \param size size of the model
            \param R2 R2 of the model
        '''

        # compute tau_max
        v = -(1 - R2) * (size + 3.0)
        a2 = (n - size - 4 - 2 * (1 - R2)) / v
        a1 = (n * (2 - R2) - 3) / v
        a0 = n / v
        q = a1 / 3.0 - a2 * a2 / 9.0
        r = (a1 * a2 - 3 * a0) / 6.0 - a2 * a2 * a2 / 27.0

        v = numpy.lib.scimath.sqrt(q * q * q + r * r)
        s1 = numpy.power(r + v, 1 / 3.0)
        s2 = numpy.power(r - v, 1 / 3.0)
        tau_max = s1 + s2 - a2 / 3.0

        # compute h(tau_max)
        h_max = 0.5 * (
                (n - size - 1) * numpy.log(1 + tau_max)
                - (n - 1) * numpy.log(1 + tau_max * (1 - R2))
                - 3 * numpy.log(tau_max)
                - n / tau_max
                )

        # compute derivative d2 h(tau_max)
        d2h_max = 0.5 * (
                ((n - 1) * (1 - R2) * (1 - R2)) / ((1 + tau_max * (1 - R2)) ** 2)
                - (n - size - 1) / ((1 + tau_max) ** 2)
                + 3.0 / (tau_max ** 2)
                - (2 * n) / (tau_max ** 3)
                )

        return -0.5 * numpy.log(-d2h_max) + h_max

    @classmethod
    def score(cls, Ystar, config, size):
        '''
            Computes the log-posterior probability of the model Ystar up to a
            constant.
            \param Ystar model
            \param config PosteriorBVS instance
            \param size size the model Ystar
        '''
        # size penalty
        L = size * config.LOGIT_P

        # compute Residual Sum of Squares
        RSS = 0.0
        if size > 0:
            C = scipy.linalg.cholesky(config.W[Ystar, :][:, Ystar])
            b = scipy.linalg.solve(C.T, config.Zty[Ystar, :])
            RSS = numpy.dot(b.T, b)

        # return Laplace approximation for prior on the dispersion parameter
        if config.PRIOR == PRIOR_ZELLNER_INVGAMMA:
            R2 = RSS / config.TSS
            L += config.zellner_invgamma_laplace(n=config.n, size=size, R2=R2)
            return L

        # compute local empirical Bayesian estimator
        if config.LOCAL_EMPIRICAL_BAYES:
            R2 = RSS / config.TSS
            F = (R2 / size) / ((1 - R2) / (config.n - 1 - size))
            tau = max(F - 1, 0.0)
            if tau > 0: shrinkage = (1 + 1 / tau)
            else: shrinkage = 1
            # re-compute constants
            config.A_TIMES_B_PLUS_SCALED_TSS = config.a * config.b + shrinkage * config.TSS
            config.NEG_HALF_LOG_ONE_PLUS_TAU = -0.5 * numpy.log(1 + tau)

        # Zellner's prior
        if config.PRIOR == PRIOR_ZELLNER:
            L += config.NEG_HALF_N_MINUS_ONE_PLUS_A * numpy.log(config.A_TIMES_B_PLUS_SCALED_TSS - RSS)
            L += config.NEG_HALF_LOG_ONE_PLUS_TAU * size

        # Independent prior
        if config.PRIOR == PRIOR_INDEPENDENT:
            L += config.NEG_HALF_N_MINUS_ONE_PLUS_A * numpy.log(config.A_TIMES_B_PLUS_TSS - RSS)
            L += config.NEG_HALF_LOG_TAU * size
            if size > 0: L -= numpy.log(C.diagonal()).sum()

        return L
