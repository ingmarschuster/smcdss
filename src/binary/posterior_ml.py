#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Criteria based on maximum likelihood like AIC and BIC.
    @namespace binary.posterior_ml
"""

import numpy
import scipy.linalg
import binary.posterior as posterior
import binary.wrapper as wrapper

class PosteriorML(posterior.Posterior):
    """ Criteria based on maximum likelihood like AIC and BIC. """

    name = 'Bayesian information criterion'

    def __init__(self, y, Z, config, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param y explained variable
            \param Z covariates to perform selection on
            \param config parameter dictionary
        """

        super(PosteriorML, self).__init__(y, Z, config, name=name, long_name=long_name)

        # add modules
        self.py_wrapper = wrapper.posterior_ml()
        self.pp_modules += ('binary.posterior_ml',)

        self.W = numpy.dot(self.Z.T, self.Z) + 1e-10 * numpy.eye(self.Z.shape[1])
        self.HALF_NEG_N = -0.5 * self.n
        LOG_P = numpy.log(config['prior/model_inclprob'])

        if config['prior/criterion'].lower() == 'bic':
            self.PENALTY = 0.5 * numpy.log(self.n) - LOG_P
        if config['prior/criterion'].lower() in ['aic', 'aicc']:
            self.PENALTY = 1 - LOG_P

        # AIC with correction
        self.AICc = False
        if config['prior/criterion'].lower() == 'aicc': self.AICc = True

    @classmethod
    def score(cls, Ystar, config, size):
        RSS = 0.0
        if size > 0:
            W_inv_Zty = scipy.linalg.solve(config.W[Ystar, :][:, Ystar], config.Zty[Ystar, :], sym_pos=True)
            RSS = numpy.dot(config.Zty[Ystar], W_inv_Zty)
        L = config.HALF_NEG_N * numpy.log(config.tss - RSS) - config.PENALTY * size

        if config.AICc: L += -size * (size + 1) / float((config.n - 1 - size))

        return L

