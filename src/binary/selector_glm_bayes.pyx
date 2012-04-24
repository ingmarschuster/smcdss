#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 
    Bayesian variable selection for generalized linear models.
    @namespace binary.selector_glm_bayes
"""

cimport numpy
import binary.selector_glm as glm
import binary.wrapper as wrapper
import numpy
import scipy.linalg
import sys

cdef extern from "math.h":
    float exp(float)
    float log(float)

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
        self.a = config['prior/var_hp_a']
        self.b = config['prior/var_hp_b']
        if self.b < 0.5:
            sys.stderr.write("\nUnfeasible prior configuration. Set b=0.5\n")
            self.b = 0.5

        # sample for MC estimate of integrated likelihood            
        self.n_samples = int(config['prior/n_samples'])

        # use Laplace approximation
        self.Laplace = False
        if config['prior/criterion'].lower() == 'laplace':
            self.Laplace = True

        self.prior = True

    def score(self, numpy.ndarray[dtype=numpy.float64_t, ndim=1] y,
                        numpy.ndarray[dtype=numpy.float64_t, ndim=2] Z,
                        numpy.ndarray[dtype=numpy.int16_t, ndim=1] index):


        cdef Py_ssize_t d = index.shape[0]
        cdef Py_ssize_t i, j
        cdef float log_proposal, log_target

        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] w = numpy.empty(self.n_samples)
        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 2] F_mle

        beta_mle, F_mle = self.compute_mle(y, Z, index)

        # return Laplace approximation of posterior
        if self.Laplace:
            return self.log_llh(y, Z, beta_mle, index) - 0.5 * log(scipy.linalg.det(F_mle, overwrite_a=True))

        # Cholesky decomposition
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


    def laplace(self, numpy.ndarray[dtype=numpy.float64_t, ndim=1] y,
                        numpy.ndarray[dtype=numpy.float64_t, ndim=2] Z,
                        numpy.ndarray[dtype=numpy.int16_t, ndim=1] index):
        """
            Laplace
            \return Laplace approximation
        """
        beta_mle, F_mle = self.compute_mle(y, Z, index)
        return self.log_llh(y, Z, beta_mle, index) - 0.5 * log(scipy.linalg.det(F_mle, overwrite_a=True))
