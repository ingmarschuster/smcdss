#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with linear conditionals. \namespace binary.qu_linear """

import numpy
cimport numpy

cdef extern from "math.h":
    double exp(double)
    double log(double)

import sys
import scipy.linalg
import binary.base
import binary.wrapper

class LinearCondBinary(binary.base.BaseBinary):
    """ Binary parametric family with linear conditionals. """

    def __init__(self, Beta):
        """
            Constructor.
            \param Beta matrix of coefficients
        """
        super(LinearCondBinary, self).__init__(d=Beta.shape[0], name='linear conditionals binary', long_name=__doc__)

        # add modules
        self.pp_modules = ('numpy', 'binary.linear_cond')

        # add dependent functions
        self.pp_depfuncs += ('_rvslpmf_all',)

        self.py_wrapper = binary.wrapper.linear_cond()

        self.Beta = Beta

    def __str__(self):
        return 'd: %d, Beta:\n%s' % (self.d, repr(self.Beta))

    @classmethod
    def _rvslpmf_all(cls, numpy.ndarray[dtype=numpy.float64_t, ndim=2] Beta,
                          numpy.ndarray[dtype=numpy.float64_t, ndim=2] U=None,
                          numpy.ndarray[dtype=numpy.int8_t, ndim=2] Y=None):
        """
            All-purpose routine for sampling and point-wise evaluation.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        cdef Py_ssize_t d = Beta.shape[0]
        cdef Py_ssize_t k, i, size
        cdef double cprob
        cdef double x

        if U is not None:
            size = U.shape[0]
            Y = numpy.empty((size, d), dtype=numpy.int8)

        if Y is not None:
            size = Y.shape[0]

        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] L = numpy.zeros(size, dtype=numpy.float64)

        for k in xrange(size):

            for i in xrange(d):
                # Compute log conditional probability that Y(i) is one
                cprob = Beta[i, i]
                for j in xrange(i): cprob += Beta[i, j] * Y[k, j]
                cprob = min(max(cprob, 0.0), 1.0)

                # Generate the ith entry
                if U is not None: Y[k, i] = U[k, i] < cprob

                # Add to log conditional probability
                if Y[k, i]:
                    if cprob == 0.0:
                        L[k] = -numpy.inf
                    else:
                        L[k] += log(cprob)
                else:
                    if cprob == 1.0:
                        L[k] = -numpy.inf
                    else:
                        L[k] += log(1.0 - cprob)

        return numpy.array(Y, dtype=bool), L


    @classmethod
    def random(cls, d, phi=0.8):
        """
            Construct a random linear model for testing.
            \param cls class
            \param d dimension
        """
        mean, corr = binary.base.moments2corr(binary.base.random_moments(d, phi=phi))
        return LinearCondBinary.from_moments(mean, corr)

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
        M = binary.base.corr2moments(mean, corr)
        d = M.shape[0]

        # Initialize Beta
        Beta = numpy.zeros((d, d), dtype=float)
        Beta[0, 0] = M[0, 0]

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

            if verbose: sys.stderr.write('V[:,:]:\n%s' % repr(V[:, index + [d]][index + [d], :]))

            Beta[i, :i + 1] = scipy.linalg.solve(V[:, index + [d]][index + [d], :], M[:i + 1, i])

            if verbose: sys.stderr.write('i: %d, beta: %s' % (i, repr(Beta[i, :i + 1])))

        if verbose: sys.stderr.write('\nlinear conditionals family successfully constructed from moments.\n\n')
        return cls(Beta)

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
    def test_properties(cls, d, n=1e4, phi=0.8, ncpus=1):
        """
            Tests functionality of the linear conditional family class.
            \param d dimension
            \param n number of samples
            \param phi dependency level in [0,1]
            \param ncpus number of cpus 
        """

        mean, corr = binary.base.moments2corr(binary.base.random_moments(d, phi=phi))
        print 'given marginals '.ljust(100, '*')
        binary.base.print_moments(mean, corr)

        generator = LinearCondBinary.from_moments(mean, corr)
        print generator.name + ':'
        print generator

        print 'formula '.ljust(100, '*')
        binary.base.print_moments(generator.mean, generator.corr)

        print 'exact \pi conditionals in [0,1] '.ljust(100, '*')
        binary.base.print_moments(generator.exact_marginals(ncpus))

        print ('simulation \pi conditionals in [0,1] (n = %d) ' % n).ljust(100, '*')
        binary.base.print_moments(generator.rvs_marginals(n, ncpus))

    def getMoments(self):

        d = self.d
        V = numpy.zeros(shape=(d + 1, d + 1), dtype=float)
        V[0, 0] = V[d, 0] = V[0, d] = self.Beta[0, 0]
        V[d, d] = 1.0

        index = list()
        # Loop over all dimensions
        for i in xrange(1, d):
            index += [i - 1]
            V[:i + 1, i] = V[i, :i + 1] = numpy.dot(V[:, index + [d]][index + [d], :], self.Beta[i, :i + 1])
            V[i, d] = V[d, i] = V[i, i]

        return V[:d, :d]

    def _getMean(self):
        """ Get expected value of instance. \return mean """
        return binary.base.moments2corr(self.getMoments())[0]

    def getCorr(self):
        """ Get correlation matrix. \return correlation matrix """
        return binary.base.moments2corr(self.getMoments())[1]

    corr = property(fget=getCorr, doc="correlation matrix")
