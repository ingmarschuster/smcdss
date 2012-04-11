#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with limited components. \namespace binary.limited """

import numpy

import scipy.special
import binary.exchangeable
import binary.wrapper

class LimitedBinary(binary.exchangeable.ExchangeableBinary):
    """ Binary parametric family limited to vectors of norm not greater than q subsets. """

    @classmethod
    def log_binomial(cls, a, b):
        return (scipy.special.gammaln(a + 1) -
                scipy.special.gammaln(b + 1) -
                scipy.special.gammaln(a - b + 1))

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

        self.py_wrapper = binary.wrapper.limited_product()

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
    def _rvs(cls, U, b):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        size, d = U.shape
        Y = numpy.zeros(U.shape, dtype=bool)

        for k in xrange(size):
            perm = numpy.arange(d)
            for i in xrange(d):
                # pick an element in p[:i+1] with which to exchange p[i]
                j = int(U[k][i] * (d - i))
                perm[d - 1 - i], perm[j] = perm[j], perm[d - 1 - i]
            # draw the number of nonzero elements

            for r, p in enumerate(b):
                if U[k][d - 1] < p: break

            Y[k][perm[:r]] = True
        return Y

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
