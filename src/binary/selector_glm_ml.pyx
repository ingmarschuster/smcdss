#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 
    Criteria based on maximum likelihood for generalized linear models.
    @namespace binary.selector_glm_ml
"""

cimport numpy
import binary.selector_glm as glm
import binary.wrapper as wrapper
import numpy

cdef extern from "math.h":
    float exp(float)
    float log(float)

class SelectorGmlMl(glm.SelectorGlm):
    """ Criteria based on maximum likelihood for generalized linear models."""

    name = 'selector glm ln'

    def __init__(self, y, Z, config, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param Y explained variable
            \param Z covariates to perform selection on
            \param config dictionary
        """

        super(SelectorGmlMl, self).__init__(y, Z, config, name=name, long_name=long_name)

        # add modules
        self.py_wrapper = wrapper.selector_glm_ml()
        self.pp_modules += ('binary.selector_glm_ml',)

        self.SIZE_PENALTY = -self.LOGIT_P

        # determine criterion
        if config['prior/criterion'].lower() == 'bic':
            self.SIZE_PENALTY += 0.5 * numpy.log(self.n)
        if config['prior/criterion'].lower() in ['aic', 'aicc']:
            self.SIZE_PENALTY += 1

        # use AIC with correction
        self.AICc = False
        if config['prior/criterion'].lower() == 'aicc':
            self.AICc = True

        self.prior = False

    def score(self, numpy.ndarray[dtype=numpy.float64_t, ndim=1] y,
                        numpy.ndarray[dtype=numpy.float64_t, ndim=2] Z,
                        numpy.ndarray[dtype=numpy.int16_t, ndim=1] index):
        """
            Criterion.
            \return criterion
        """
        size = index.shape[0] - 1
        beta_mle, F_mle = self.compute_mle(y, Z, index)
        L = self.log_llh(y, Z, beta_mle, index) - self.SIZE_PENALTY * size
        if self.AICc: L -= size * (size + 1) / float((self.n - 1 - size))
        return L
