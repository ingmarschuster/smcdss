#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with constrained components. \namespace binary.product_constrained """

import numpy
cimport numpy

import product
import binary.wrapper as wrapper

class ConstrProductBinary(product.ProductBinary):
    """ Binary parametric family constrained to vectors that verify certain interactions. """

    def __init__(self, p, constrained=None, name='constrained product family', long_name=__doc__):
        """ 
            Constructor.
            \param d dimension
            \param constraints list
            \param name name
            \param long_name long_name
        """

        super(ConstrProductBinary, self).__init__(p=p, name=name, long_name=long_name)

        # add module
        self.py_wrapper = wrapper.product_constrained()
        self.pp_modules += ('binary.product_constrained',)

        ## list of entries having interaction constraints
        self.constrained = constrained
        
        ## list of free entries
        self.free = numpy.array([i for i in xrange(self.d) if not i in [item for sublist in constrained for item in sublist]], dtype=numpy.int16)

    def __str__(self):
        return 'd: %d, p:\n%sconstraints:\n%s\n' % (self.d, repr(self.p), repr(self.constrained))

    @classmethod
    def from_moments(cls, mean, corr=None):
        """ 
            Construct a random family for testing.
            \param mean mean vector
        """
        return cls.random(d=mean.shape[0], p=mean)

    @classmethod
    def _lpmf(cls, Y, p, constrained):
        """ 
            Log-probability mass function.
            \param gamma binary vector
            \param param parameters
            \return log-probabilities
        """
        L = product.ProductBinary._lpmf(numpy.array(Y, dtype=numpy.int8), p)
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
    def _rvs(cls, numpy.ndarray[dtype=numpy.float64_t, ndim=2] U,
                  numpy.ndarray[dtype=numpy.float64_t, ndim=1] p,
                  constrained,
                  numpy.ndarray[dtype=numpy.int16_t, ndim=1] free):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        
        cdef Py_ssize_t k, i
        cdef double p_all
        cdef numpy.ndarray[dtype = Py_ssize_t, ndim = 1] bundle
        cdef numpy.ndarray[dtype = numpy.int8_t, ndim = 2] Y = numpy.zeros((U.shape[0], U.shape[1]), dtype=numpy.int8)

        for k in xrange(U.shape[0]):
            for i in free: Y[k, i] = U[k, i] < p[i]
            for bundle in constrained:
                p_all = 1.0
                for i in bundle: p_all *= p[i]
                if U[k, bundle[0]] < p_all / (1.0 - p[bundle[0]] + p_all):
                    for i in bundle: Y[k, i] = True
                else:
                    for i in bundle: Y[k, i] = U[k, i] < p[i]
                    Y[k, bundle[0]] = False

        return numpy.array(Y, dtype=bool)

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
