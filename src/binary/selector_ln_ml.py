#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Criteria based on maximum likelihood for linear normal models.
    @namespace binary.selector_ln_ml
"""

import numpy
import scipy.linalg
import binary.selector_ln as ln
import binary.wrapper as wrapper

class SelectorLnMl(ln.SelectorLn):
    """ Criteria based on maximum likelihood for linear normal models."""

    name = 'selector ln ml'

    def __init__(self, y, Z, config, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param y explained variable
            \param Z covariates to perform selection on
            \param config parameter dictionary
        """

        super(SelectorLnMl, self).__init__(y, Z, config, name=name, long_name=long_name)

        # add modules
        self.py_wrapper = wrapper.selector_ln_ml()
        self.pp_modules += ('binary.selector_ln_ml',)

        self.W = numpy.dot(self.Z.T, self.Z) + 1e-10 * numpy.eye(self.Z.shape[1])
        self.HALF_NEG_N = -0.5 * self.n

        if config['prior/criterion'].lower() == 'bic':
            self.SIZE_PENALTY = 0.5 * numpy.log(self.n) - self.LOGIT_P
        if config['prior/criterion'].lower() in ['aic', 'aicc']:
            self.SIZE_PENALTY = 1 - self.LOGIT_P

        # AIC with correction
        self.AICc = False
        if config['prior/criterion'].lower() == 'aicc': self.AICc = True

    @classmethod
    def score(cls, Ystar, config, size):
        RSS = 0.0
        if size > 0:
            W_inv_Zty = scipy.linalg.solve(config.W[Ystar, :][:, Ystar], config.Zty[Ystar, :], sym_pos=True)
            RSS = numpy.dot(config.Zty[Ystar], W_inv_Zty)
        L = config.HALF_NEG_N * numpy.log(config.tss - RSS) - config.SIZE_PENALTY * size

        if config.AICc: L += -size * (size + 1) / float((config.n - 1 - size))

        return L

