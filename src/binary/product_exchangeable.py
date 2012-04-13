#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with exchangeable components. \namespace binary.product_exchangeable """

import numpy

import binary.base as base
import binary.wrapper as wrapper

class ExchangeableBinary(base.BaseBinary):
    """ Binary parametric family with exchangeable components."""

    def __init__(self, d, p, name='exchangeable product family', long_name=__doc__):
        """ 
            Constructor.
            \param p mean vector
            \param name name
            \param long_name long_name
        """

        # call super constructor
        super(ExchangeableBinary, self).__init__(d=d, name=name, long_name=long_name)

        # add module
        self.py_wrapper = wrapper.product_exchangeable()
        self.pp_modules += ('binary.product_exchangeable',)

        self.p = p
        self.logit_p = numpy.log(p / (1.0 - p))

    def __str__(self):
        return 'd: %d, p: %.4f' % (self.d, self.p)

    @classmethod
    def from_moments(cls, mean, corr=None):
        """ 
            Construct a random family for testing.
            \param mean mean vector
        """
        return cls(d=mean.shape[0], p=mean[0])

    @classmethod
    def _lpmf(cls, Y, logit_p):
        """ 
            Log-probability mass function.
            \param Y binary vector
            \param param parameters
            \return log-probabilities
        """
        return Y.sum(axis=1) * logit_p

    @classmethod
    def _rvs(cls, U, p):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        return U < p

    @classmethod
    def random(cls, d):
        """ 
            Construct a random family for testing.
            \param d dimension
        """
        return cls(d=d, p=0.01 + numpy.random.random() * 0.98)

    @classmethod
    def uniform(cls, d):
        """ 
            Construct a random product model for testing.
            \param d dimension
        """
        return cls(d=d, p=0.5)

    def _getMean(self):
        """ Get expected value of instance. \return p-vector """
        return self.param['p'] * numpy.ones(self.param['d'])

    def _getRandom(self, eps=0.0):
        return range(self.d)
