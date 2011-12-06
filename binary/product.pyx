#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with independent components."""

"""
\namespace binary.product
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

import cython
import numpy
cimport numpy

import utils
import binary.base
import binary.wrapper


def _lpmf(numpy.ndarray[dtype=numpy.int8_t, ndim=2] gamma,
          numpy.ndarray[dtype=numpy.float64_t, ndim=1] p):
    """ 
        Log-probability mass function.
        \param gamma binary vector
        \param param parameters
        \return log-probabilities
    """
    cdef int k
    cdef double prob, m

    L = numpy.empty(gamma.shape[0])
    for k in xrange(gamma.shape[0]):
        prob = 1.0
        for i, m in enumerate(p):
            if gamma[k, i]: prob *= m
            else: prob *= (1 - m)
        L[k] = prob
    return numpy.log(L)

def _rvs(U, param):
    """ 
        Generates a random variable.
        \param U uniform variables
        \param param parameters
        \return binary variables
    """
    p = param['p']
    Y = numpy.empty((U.shape[0], U.shape[1]), dtype=bool)
    for k in xrange(U.shape[0]):
        Y[k] = p > U[k]
    return Y


class ProductBinary(binary.base.BaseBinary):
    """ Binary parametric family with independent components."""

    def __init__(self, p=None, py_wrapper=None, name='product family', long_name=__doc__):
        """ 
            Constructor.
            \param p mean vector
            \param name name
            \param long_name long_name
        """

        if cython.compiled: print "Yep, I'm compiled."
        else: print "Just a lowly interpreted script."

        # link to python wrapper
        if py_wrapper is None: py_wrapper = binary.wrapper.product()

        # call super constructor
        binary.base.BaseBinary.__init__(self, py_wrapper, name=name, long_name=long_name)

        # add module
        self.pp_modules += ('binary.product',)

        if not p is None:
            if isinstance(p, (numpy.ndarray, list)):
                p = numpy.array(p, dtype=float)
            else:
                p = numpy.array([p])

        self.param.update({'p':p})

    def __str__(self):
        return utils.format.format_vector(self.p, 'p')

    @classmethod
    def random(cls, d):
        """ 
            Construct a random product model for testing.
            \param cls class
            \param d dimension
        """
        return cls(numpy.random.random(d))

    @classmethod
    def uniform(cls, d):
        """ 
            Construct a random product model for testing.
            \param cls class
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

    def renew_from_data(self, sample, lag=0.0, verbose=False):
        """ 
            Updates the product model from data.
            \param cls class
            \param sample a sample of binary data
            \param lag lag
            \param verbose detailed information
        """
        p = sample.getMean(weight=True)
        self.param['p'] = (1 - lag) * p + lag * self.p

    def getP(self):
        """ Get p-vector. \return p-vector """
        return self._getP()

    def _getP(self):
        """ Get p-vector of instance. \return p-vector """
        return self.param['p']

    def _getMean(self):
        """ Get expected value of instance. \return p-vector """
        return self.param['p']

    def _getD(self):
        """ Get dimension of instance. \return dimension """
        return self.p.shape[0]

    def _getRandom(self, xi=binary.base.BaseBinary.MIN_MARGINAL_PROB):
        """ Get index list of random components of instance. \return index list """
        return [i for i, p in enumerate(self.param['p']) if min(p, 1.0 - p) > xi]

    p = property(fget=getP, doc="p")
