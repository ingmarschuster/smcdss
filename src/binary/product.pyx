#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with independent components. \namespace binary.product """

import numpy
cimport numpy

import product_exchangeable
import binary.base as base
import binary.wrapper as wrapper

class ProductBinary(product_exchangeable.ExchangeableBinary):
    """ Binary parametric family with independent components."""

    name = 'product family'

    def __init__(self, p, name=name, long_name=__doc__):
        """
            Constructor.
            \param p mean vector
            \param name name
            \param long_name long_name
        """

        if isinstance(p, float): p = [p]
        p = numpy.array(p, dtype=float)

        # call super constructor
        super(ProductBinary, self).__init__(d=p.shape[0], p=0.5, name=name, long_name=long_name)

        # add module
        self.py_wrapper = wrapper.product()
        self.pp_modules += ('binary.product',)

        self.p = p

    def __str__(self):
        return 'd: %d, p:\n%s' % (self.d, repr(self.p))

    @classmethod
    def _lpmf(cls,
              numpy.ndarray[dtype=numpy.int8_t, ndim=2] Y,
              numpy.ndarray[dtype=numpy.float64_t, ndim=1] p):
        """ 
            Log-probability mass function.
            \param gamma binary vector
            \param param parameters
            \return log-probabilities
        """
        cdef int k
        cdef double prob, m

        L = numpy.empty(Y.shape[0])
        for k in xrange(Y.shape[0]):
            prob = 1.0
            for i, m in enumerate(p):
                if Y[k, i]: prob *= m
                else: prob *= (1 - m)
            L[k] = prob
        return numpy.log(L)

    @classmethod
    def _rvs(cls, U, p):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        Y = numpy.empty((U.shape[0], U.shape[1]), dtype=bool)
        for k in xrange(U.shape[0]):
            Y[k] = p > U[k]
        return Y

    @classmethod
    def from_moments(cls, mean, corr=None):
        """ 
            Construct a random family for testing.
            \param mean mean vector
        """
        return cls(p=mean)

    @classmethod
    def random(cls, d):
        """ 
            Construct a random family for testing.
            \param d dimension
        """
        return cls(0.01 + numpy.random.random(d) * 0.98)

    @classmethod
    def uniform(cls, d):
        """ 
            Construct a uniform.
            \param d dimension
        """
        return cls(p=0.5 * numpy.ones(d))

    @classmethod
    def from_data(cls, sample):
        """ 
            Construct a product model from data.
            \param cls class
            \param sample a sample of binary data
        """
        return cls(sample.mean)

    def renew_from_data(self, X, weights, lag=0.0, verbose=False):
        """ 
            Updates the product model from data.
            \param cls class
            \param sample a sample of binary data
            \param lag lag
            \param verbose detailed information
        """
        p = base.sample2mean(X, weights=None)
        self.p = (1.0 - lag) * p + lag * self.p

    def _getMean(self):
        """ Get expected value of instance. \return p-vector """
        return self.p

    def _getRandom(self, xi=base.BaseBinary.MIN_MARGINAL_PROB):
        """ Get index list of random components of instance. \return index list """
        return [i for i, p in enumerate(self.p) if min(p, 1.0 - p) > xi]
