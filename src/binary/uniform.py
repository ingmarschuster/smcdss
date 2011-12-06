#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family uniform on certain subsets. """

"""
@namespace binary.uniform
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

import numpy
import math

import binary.base
import binary.wrapper
import scipy.special

def log_binomial(a, b):
    return (scipy.special.gammaln(a + 1) -
            scipy.special.gammaln(b + 1) -
            scipy.special.gammaln(a - b + 1))

class UniformBinary(binary.base.BaseBinary):
    """ Binary parametric family uniform on certain subsets. """

    def __init__(self, d, q, py_wrapper=None, name='uniform family', long_name=__doc__):
        """ 
            Constructor.
            \param d dimension
            \param q maximum size
            \param name name
            \param long_name long_name
        """

        if py_wrapper is None: py_wrapper = binary.wrapper.uniform()

        binary.base.BaseBinary.__init__(self, py_wrapper=py_wrapper, name=name, long_name=long_name)

        #n_feasible = 2 ** d - math.exp(log_binomial(d, q + 1)) * scipy.special.hyp2f1(1, q + 1 - d, q + 2, -1)

        # compute multinomial
        m = numpy.empty(q + 1, dtype=float)
        for k in xrange(q + 1):
            m[k] = log_binomial(d, k)

        # deal with sum of exponentials
        v = m.max()
        m = numpy.exp(m - v)
        p = m.sum()
        m /= p
        log_p = numpy.log(p) + v

        self.param.update({'d':d, 'q':q, 'm':m.cumsum(), 'log_p':log_p})
        self.pp_modules = ('numpy', 'binary.uniform')

    def __str__(self):
        return 'maximum size: %d' % self.param['q']

    @classmethod
    def _lpmf(cls, gamma, param):
        """ 
            Log-probability mass function.
            \param gamma binary vector
            \param param parameters
            \return log-probabilities
        """
        L = -numpy.inf * numpy.ones(gamma.shape[0])
        L[gamma.sum(axis=1) <= param['q']] = param['log_p']
        return L

    @classmethod
    def _rvs(cls, U, param):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        Y = numpy.zeros((U.shape[0], U.shape[1]), dtype=bool)
        d, m = param['d'], param['m']

        for k in xrange(U.shape[0]):
            perm = numpy.arange(d)
            for i in xrange(d):
                # pick an element in p[:i+1] with which to exchange p[i]
                j = int(U[k][i] * (d - i))
                perm[d - 1 - i], perm[j] = perm[j], perm[d - 1 - i]
            # draw the number of nonzero elements
            for r, p in enumerate(m):
                if U[k][d - 1] < p: break
            Y[k][perm[:r]] = True
        return Y

    @classmethod
    def random(cls, d):
        """ 
            Construct a random product model for testing.
            \param cls class
            \param d dimension
        """
        return cls(d, numpy.random.randint(d) + 1)

    def _getMean(self):
        mean = 0.5 * self.param['q'] / float(self.param['d'])
        return mean * numpy.ones(self.param['d'])

    def _getD(self):
        """ Get dimension of instance. \return dimension """
        return self.param['d']
