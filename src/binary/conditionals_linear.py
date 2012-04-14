#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with linear conditionals. \namespace binary.conditionals_linear """

import numpy

import scipy.linalg
import binary.conditionals as conditionals
import binary.base as base
import binary.wrapper as wrapper

class LinearCondBinary(conditionals.ConditionalsBinary):
    """ Binary parametric family with linear conditionals. """

    def __init__(self, A, name='linear conditionals family', long_name=__doc__):
        """ 
            Constructor.
            \param A Lower triangular matrix holding regression coefficients
            \param name name
            \param long_name long name
        """

        # call super constructor
        super(LinearCondBinary, self).__init__(A=A, name=name, long_name=long_name)

        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg', 'binary.conditionals_linear',)

        self.py_wrapper = wrapper.conditionals_linear()

    @classmethod
    def from_moments(cls, mean, corr, verbose=False):
        """
            Constructs a linear model for given moments. Warning: This method
            might produce parameters that are infeasible and yield an improper
            distribution.
            \param mean mean
            \param corr correlation matrix
        """

        # Convert arguments
        M = base.corr2moments(mean, corr)
        d = M.shape[0]

        # Initialize A
        A = numpy.zeros((d, d), dtype=float)
        A[0, 0] = M[0, 0]

        # Create auxiliary matrix
        V = numpy.empty((d + 1, d + 1), dtype=float)
        V[:d, :d] = M
        V[d, :-1] = V[:-1, d] = M.diagonal()
        V[d, d] = 1.0
        V += 1e-5 * numpy.eye(d + 1)

        index = list()
        # Loop over all dimensions
        for i in xrange(1, d):
            index += [i - 1]
            A[i, :i + 1] = scipy.linalg.solve(V[:, index + [d]][index + [d], :], M[:i + 1, i])

        return cls(A)

    @classmethod
    def from_data(cls, sample):
        """
            Constructs a linear model from data. Warning: This method might
            produce parameters that are infeasible and yield an improper
            distribution.
            \param d dimension
        """
        return cls.from_moments(sample.mean, sample.cor)

    @classmethod
    def link(cls, x):
        """ Link function """
        return numpy.maximum(0, numpy.minimum(x, 1))

    @classmethod
    def ilink(cls, p):
        """ Inverse of link function """
        return p

    def exact_marginals(self, ncpus=None):
        """ Get mean and correlation matrix. \return mean and correlation matrix """
        d = self.d
        V = numpy.zeros(shape=(d + 1, d + 1), dtype=float)
        V[0, 0] = V[d, 0] = V[0, d] = self.A[0, 0]
        V[d, d] = 1.0

        index = list()
        for i in xrange(1, d):
            index += [i - 1]
            V[:i + 1, i] = V[i, :i + 1] = numpy.dot(V[:, index + [d]][index + [d], :], self.A[i, :i + 1])
            V[i, d] = V[d, i] = V[i, i]

        return base.moments2corr(V[:d, :d])

    def _getMean(self):
        """ Get expected value of instance. \return mean """
        return self.exact_marginals()[0]

    def getCorr(self):
        """ Get correlation matrix. \return correlation matrix """
        return self.exact_marginals()[1]

    corr = property(fget=getCorr, doc="correlation matrix")
