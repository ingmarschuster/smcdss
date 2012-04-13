#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with limited components. \namespace binary.limited """

import numpy
cimport numpy

import scipy.special
import product_exchangeable
import binary.wrapper as wrapper

class LimitedBinary(product_exchangeable.ExchangeableBinary):
    """ Binary parametric family limited to vectors of norm not greater than q subsets. """

    def __init__(self, d, q, p=0.5, name='limited product family', long_name=__doc__):
        """ 
            Constructor.
            \param d dimension
            \param q maximum size
            \param p marginal probability
            \param name name
            \param long_name long_name
        """

        super(LimitedBinary, self).__init__(d=d, p=p, name=name, long_name=long_name)

        # add module
        self.py_wrapper = wrapper.product_limited()
        self.pp_modules += ('binary.product_limited',)

        # compute binomial probabilities up to q
        b = numpy.empty(q + 1, dtype=float)
        for k in xrange(q + 1):
            b[k] = LimitedBinary.log_binomial(d, k) + k * self.logit_p

        # deal with sum of exponentials
        b = numpy.exp(b - b.max())
        b /= b.sum()

        self.q = q
        self.b = b.cumsum()

    def __str__(self):
        return 'd: %d, limit: %d, p: %.4f\n' % (self.d, self.q, self.p)

    @classmethod
    def from_moments(cls, mean, corr=None):
        """ 
            Construct a random family for testing.
            \param mean mean vector
        """
        return cls.random(d=mean.shape[0], p=mean[0])

    @classmethod
    def log_binomial(cls, a, b):
        return (scipy.special.gammaln(a + 1) - 
                scipy.special.gammaln(b + 1) - 
                scipy.special.gammaln(a - b + 1))

    @classmethod
    def _lpmf(cls, Y, q, logit_p):
        """ 
            Log-probability mass function.
            \param gamma binary vector
            \param param parameters
            \return log-probabilities
        """
        L = -numpy.inf * numpy.ones(Y.shape[0])
        index = numpy.array(Y.sum(axis=1), dtype=float)
        index = index[index <= q]
        L[Y.sum(axis=1) <= q] = index * logit_p
        return L

    @classmethod
    def _rvs(cls, numpy.ndarray[dtype=numpy.float64_t, ndim=2] U,
                  numpy.ndarray[dtype=numpy.float64_t, ndim=1] b):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        cdef Py_ssize_t size = U.shape[0]
        cdef Py_ssize_t d = U.shape[1]
        cdef Py_ssize_t i, j
        
        cdef numpy.ndarray[Py_ssize_t, ndim = 1] perm = numpy.arange(d, dtype=numpy.int)
        cdef numpy.ndarray[dtype = numpy.int8_t, ndim = 2] Y = numpy.zeros((U.shape[0], U.shape[1]), dtype=numpy.int8)
        
        for k in xrange(size):
            perm = numpy.arange(d, dtype=numpy.int)
            for i in xrange(d):
                # pick an element in p[:i+1] with which to exchange p[i]
                j = int(U[k, i] * (d - i))
                perm[d - 1 - i], perm[j] = perm[j], perm[d - 1 - i]
            
            # draw the number of nonzero elements
            for r in xrange(b.shape[0]):
                if U[k, d - 1] < b[r]: break

            for i in xrange(r): Y[k, perm[i]] = True

        return numpy.array(Y, dtype=bool)

    @classmethod
    def random(cls, d, p=None):
        """ 
            Construct a random family for testing.
            \param cls class
            \param d dimension
        """
        # random marginal probability
        if p is None: p = 0.01 + 0.98 * numpy.random.random()

        # random maximal norm
        q = numpy.random.randint(d) + 1
        return cls(d=d, q=q, p=p)

    def _getMean(self):
        mean = self.p * self.q / float(self.d)
        return mean * numpy.ones(self.d)
