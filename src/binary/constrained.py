#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with constrained components. \namespace binary.constrained """

import numpy

import product
import wrapper

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

        self.py_wrapper = wrapper.constrained_product()

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
