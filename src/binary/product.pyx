#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with independent components."""

"""
\namespace binary.product
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

import numpy
cimport numpy

import utils
import scipy.special
import binary.base
import binary.wrapper


class EquableProductBinary(binary.base.BaseBinary):
    """ Binary parametric family with independent components."""

    def __init__(self, d, p, name='equable product family', long_name=__doc__):
        """ 
            Constructor.
            \param p mean vector
            \param name name
            \param long_name long_name
        """

        # call super constructor
        binary.base.BaseBinary.__init__(self, d=d, name=name, long_name=long_name)

        # add module
        self.py_wrapper = binary.wrapper.equable_product()
        self.pp_modules += ('binary.product',)

        self.p = p
        self.logit_p = numpy.log(p / (1.0 - p))

    def __str__(self):
        return 'd: %d, p: %.4f' % (self.d, self.p)

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



class ProductBinary(EquableProductBinary):
    """ Binary parametric family with independent components."""

    def __init__(self, p, name='product family', long_name=__doc__):
        """
            Constructor.
            \param p mean vector
            \param name name
            \param long_name long_name
        """

        if isinstance(p, float): p = [p]
        p = numpy.array(p, dtype=float)

        # call super constructor
        EquableProductBinary.__init__(self, d=p.shape[0], p=0.5, name=name, long_name=long_name)

        self.py_wrapper = binary.wrapper.product()

        # add module
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
    def from_moments(cls, mean, corr):
        """ 
            Construct a random family for testing.
            \param mean mean vector
            \param corr correlation matrix (nuisance parameter)
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

    def _getMean(self):
        """ Get expected value of instance. \return p-vector """
        return self.p

    def _getRandom(self, xi=binary.base.BaseBinary.MIN_MARGINAL_PROB):
        """ Get index list of random components of instance. \return index list """
        return [i for i, p in enumerate(self.p) if min(p, 1.0 - p) > xi]





class PositiveProductBinary(ProductBinary):
    """ Binary parametric family with independent components and positive support."""

    def __init__(self, p, name='positive product family', long_name=__doc__):
        """ 
            Constructor.
            \param p mean vector
            \param name name
            \param long_name long name
        """

        # call super constructor
        binary.product.ProductBinary.__init__(self, p=p, name=name, long_name=long_name)

        # add dependent functions
        self.pp_depfuncs += ('_rvslpmf_all',)
        self.py_wrapper = binary.wrapper.positive_product()

        log_c = numpy.log(1.0 - numpy.cumprod(1.0 - p[::-1])[::-1]) # prob of gamma > 0 at component 1,..,d
        log_c = numpy.append(log_c, -numpy.inf)

        self.log_p = numpy.log(self.p)
        self.log_q = numpy.log(1.0 - self.p)
        self.log_c = log_c

    def __str__(self):
        return 'd: %d, p:\n%smean:\n%s' % (self.d, repr(self.p), repr(self.mean))

    @classmethod
    def _rvslpmf_all(cls, log_p, log_q, log_c, U=None, Y=None):
        """ 
            All-purpose routine for sampling and point-wise evaluation.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        cdef int size, d, k, i
        if U is not None:
            size, d = U.shape
            Y = numpy.empty((size, d), dtype=bool)
            logU = numpy.log(U)

        if Y is not None:
            size, d = Y.shape

        L = numpy.zeros(size, dtype=float)

        for k in xrange(size):

            for i in xrange(d):
                # Compute log conditional probability that gamma(i) is one
                logcprob = log_p[i]
                if not Y[k, :i].any(): logcprob -= log_c[i]

                # Generate the ith entry
                if U is not None: Y[k, i] = logU[k, i] < logcprob

                # Add to log conditional probability
                if Y[k, i]:
                    L[k] += logcprob
                else:
                    L[k] += log_q[i]
                    if not Y[k, :i].any(): L[k] += log_c[i + 1] - log_c[i]

        return Y, L

    @classmethod
    def random(cls, d):
        """ 
            Construct a random product model for testing.
            \param cls class
            \param d dimension
        """
        return cls(0.01 + numpy.random.random(d) * 0.1)

    def _getMean(self):
        return numpy.exp(self.log_p - self.log_c[0])





class ConstrProductBinary(ProductBinary):
    """ Binary parametric family constrained to vectors that verify certain interactions. """

    def __init__(self, p, constrained=None, name='constrained product family', long_name=__doc__):
        """ 
            Constructor.
            \param d dimension
            \param constraints list
            \param name name
            \param long_name long_name
        """

        ProductBinary.__init__(self, p=p, name=name, long_name=long_name)

        self.py_wrapper = binary.wrapper.constrained_product()

        ## list of entries having interaction constraints
        self.constrained = constrained
        ## list of free entries
        self.free = [i for i in xrange(self.d) if not i in [item for sublist in constrained for item in sublist]]

    def __str__(self):
        return 'd: %d, p:\n%sconstraints:\n%s\n' % (self.d, repr(self.p), repr(self.constrained))

    @classmethod
    def _lpmf(cls, Y, p, constrained):
        """ 
            Log-probability mass function.
            \param gamma binary vector
            \param param parameters
            \return log-probabilities
        """
        L = ProductBinary._lpmf(numpy.array(Y, dtype=numpy.int8), p)
        violations = ConstrProductBinary.constraint_violation(Y, p, constrained)
        L[violations] = -numpy.inf
        return L

    @classmethod
    def constraint_violation(cls, Y, p, constrained):
        V = numpy.zeros(Y.shape[0], dtype=bool)

        for k in xrange(Y.shape[0]):
            for bundle in constrained:
                if Y[k, bundle[0]] and not Y[k, bundle[1:]].prod():
                    V[k] = True
                    break
        return V

    @classmethod
    def _rvs(cls, U, p, constrained, free):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        Y = numpy.zeros((U.shape[0], U.shape[1]), dtype=bool)

        for k in xrange(U.shape[0]):
            Y[k][free] = U[k][free] < p[free]
            for bundle in constrained:
                index = list(bundle)
                p_all = p[bundle].prod()
                if U[k, bundle[0]] < p_all / (1.0 - p[bundle[0]] + p_all):
                    Y[k][index] = numpy.ones(len(bundle), dtype=bool)
                else:
                    Y[k][index] = U[k][index] < p[index]
                    Y[k, bundle[0]] = False
        return Y

    @classmethod
    def random(cls, d, p=None):
        """ 
            Construct a random family for testing.
            \param cls class
            \param d dimension
        """
        # random marginal probability
        if p is None: p = 0.01 + 0.98 * numpy.random.random(d)
        if isinstance(p, float): p = p * numpy.ones(d)

        perm = numpy.arange(d, dtype=int)
        # random permutation
        for i in xrange(d):
            j = int(numpy.random.random() * (d - i))
            perm[d - 1 - i], perm[j] = perm[j], perm[d - 1 - i]
        constrained = []
        # random constraints
        i = 0
        while True:
            n = numpy.random.randint(low=2, high=5)
            if i + n > d:break
            constrained += [(numpy.array(perm[i:(i + n)]))]
            i += n
        return cls(p=p, constrained=constrained)




class LimitedProductBinary(EquableProductBinary):
    """ Binary parametric family constrained to vectors of norm not greater than q subsets. """

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

        EquableProductBinary.__init__(self, d=d, p=p, name=name, long_name=long_name)

        py_wrapper = binary.wrapper.limited_product()

        # compute binomial probabilities up to q
        b = numpy.empty(q + 1, dtype=float)
        for k in xrange(q + 1):
            b[k] = LimitedProductBinary.log_binomial(d, k) + k * self.logit_p

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
