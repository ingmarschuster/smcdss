#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Schwarz's criterion for Bayesian variable selection.
    @namespace binary.posterior_bic
"""

import numpy
import scipy.linalg
import binary.posterior as posterior
import binary.wrapper as wrapper

class PosteriorBIC(posterior.Posterior):
    """ Schwarz's criterion for Bayesian variable selection."""

    name = 'Bayesian information criterion'

    def __init__(self, y, Z, config, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param y explained variable
            \param Z covariates to perform selection on
            \param config parameter dictionary
        """

        super(PosteriorBIC, self).__init__(y, Z, config, name=name, long_name=long_name)

        # add modules
        self.py_wrapper = wrapper.posterior_bic()
        self.pp_modules += ('binary.posterior_bic',)

        self.W = numpy.dot(self.Z.T, self.Z) + 1e-10 * numpy.eye(self.Z.shape[1])
        self.HALF_NEG_N = -0.5 * self.n
        self.HALF_NEG_LOG_N = -0.5 * numpy.log(self.n)

    @classmethod
    def score(cls, Ystar, config, size):
        btb = 0.0
        if size > 0:
            W_inv_Zty = scipy.linalg.solve(config.W[Ystar, :][:, Ystar], config.Zty[Ystar, :], sym_pos=True)
            btb = numpy.dot(config.Zty[Ystar], W_inv_Zty)
        return config.HALF_NEG_N * numpy.log(config.tss - btb) + config.HALF_NEG_LOG_N * size