#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with logistic conditionals. \namespace binary.conditonal"""

import numpy
cimport numpy

cdef extern from "math.h":
    double exp(double)
    double log(double)

import scipy.linalg
import sys
import time
import utils

import product
import base
import wrapper

class ConditionalsBinary(product.ProductBinary):
    """ Binary parametric family with glm conditionals. """

    PRECISION = base.BaseBinary.PRECISION
    MAX_ENTRY_SUM = numpy.finfo(float).maxexp * log(2)

    def __init__(self, A, name='conditionals family', long_name=__doc__):
        """ 
            Constructor.
            \param A Lower triangular matrix holding regression coefficients
            \param name name
            \param long_name long name
        """

        p = ConditionalsBinary.link(numpy.diagonal(A))

        # call super constructor
        super(ConditionalsBinary, self).__init__(p=p, name=name, long_name=long_name)

        self.py_wrapper = wrapper.conditionals()

        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg',)

        # add dependent functions
        self.pp_depfuncs += ('_rvslpmf_all',)

        self.A = A

    def __str__(self):
        return 'd: %d, A:\n%s' % (self.d, repr(self.A))

    @classmethod
    def _rvslpmf_all(cls, numpy.ndarray[dtype=numpy.float64_t, ndim=2] A,
                          numpy.ndarray[dtype=numpy.float64_t, ndim=2] U=None,
                          numpy.ndarray[dtype=numpy.int8_t, ndim=2] Y=None):
        """
            All-purpose routine for sampling and point-wise evaluation.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        cdef Py_ssize_t d = A.shape[0]
        cdef Py_ssize_t k, i, size
        cdef double logprob
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
                x = A[i, i]
                for j in xrange(i):
                    x += A[i, j] * Y[k, j]
                logcprob = log(ConditionalsBinary.link(x))

                # Generate the ith entry
                if U is not None: Y[k, i] = log(U[k, i]) < logcprob

                # Add to log conditional probability
                L[k] += logcprob
                if not Y[k, i]: L[k] -= x

        return numpy.array(Y, dtype=bool), L

    @classmethod
    def independent(cls, p):
        """
            Constructs a logistic binary model with independent components.
            \param cls instance
            \param p mean
            \return logistic model
        """
        A = numpy.diag(ConditionalsBinary.ilink(p))
        return cls(A)

    @classmethod
    def uniform(cls, d):
        """ 
            Constructs a uniform logistic binary model.
            \param cls instance
            \param d dimension
            \return logistic model
        """
        A = numpy.zeros((d, d))
        return cls(A)

    @classmethod
    def random(cls, d, dep=3.0):
        """ 
            Constructs a random logistic binary model.
            \param cls instance
            \param d dimension
            \param dep strength of dependencies [0,inf)
            \return logistic model
        """
        cls = ConditionalsBinary.independent(p=numpy.random.random(d))
        A = numpy.random.normal(scale=dep, size=(d, d))
        A *= numpy.dot(A, A)
        for i in xrange(d): A[i, i] = cls.A[i, i]
        cls.A = A
        return cls

    @classmethod
    def from_moments(cls, mean, corr, n=1e4, q=25.0, delta=0.005, verbose=False):
        """ 
            Constructs a logistic conditionals family from given mean and correlation.
            \param mean mean
            \param corr correlation
            \param n number of samples for Monte Carlo estimation
            \param q number of intermediate steps in Newton-Raphson procedure
            \param delta minimum absolute value of correlation coefficients
            \return logistic conditionals family
        """

        ## dimension of binary family
        cdef Py_ssize_t d = mean.shape[0]

        ## dimension of the current logistic regression
        cdef Py_ssize_t c

        ## dimension of the sparse logistic regression
        cdef Py_ssize_t s

        ## iterators
        cdef Py_ssize_t k, i, j

        ## minimum dimension for Monte Carlo estimates
        cdef Py_ssize_t min_c = int(numpy.log2(n))

        ## probability of binary vector
        cdef double prob

        ## floating point variable
        cdef double x, high, low

        ## parameter matrix holding regression coefficients
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] A = numpy.zeros((d, d), dtype=numpy.float64)

        ## parameter vector holding regression coefficients
        cdef numpy.ndarray[numpy.float64_t, ndim = 1] a = numpy.empty(0, dtype=numpy.float64)

        ## f: mapping of the parameter vector onto the cross-moment vector
        cdef numpy.ndarray[numpy.float64_t, ndim = 1] f = numpy.empty(0, dtype=numpy.float64)

        ## Jacobian matrix of f
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] J = numpy.empty((0, 0), dtype=numpy.float64)

        ## probability mass for enumeration 
        cdef numpy.ndarray[numpy.float64_t, ndim = 1] pm = numpy.empty(0, dtype=numpy.float64)

        ## array holding uniform [0,1] random variables
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] U = numpy.empty((0, 0), dtype=numpy.float64)

        ## array holding random binary vectors
        cdef numpy.ndarray[numpy.int8_t, ndim = 2] Y = numpy.empty((0, 0), dtype=numpy.int8)

        ## index vector for sparse logistic regression
        cdef numpy.ndarray[Py_ssize_t, ndim = 1] S = numpy.empty(0, dtype=numpy.int)

        ## index vector for intermediate steps in Newton-Raphson procedure
        cdef numpy.ndarray[numpy.float64_t, ndim = 1] Q = numpy.linspace(1 / (float(q) - 1), 1.0, q - 1)

        ## cross-moment matrix
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] M = base.corr2moments(mean, corr)

        ## cross-moment matrix for independent components
        cdef numpy.ndarray[numpy.float64_t, ndim = 2] I = M.copy()

        ## parameter for convex combination between target and independent moment vectors 
        cdef double phi



        # compute cross-moment for independent case
        for i in xrange(d):
            for j in xrange(i):
                I[i, j] = M[i, i] * M[j, j]
                I[j, i] = I[i, j]

        # initialize for component of A
        A[0, 0] = ConditionalsBinary.ilink(M[0, 0])

        # loop over dimensions
        for c in xrange(1, d):
            if verbose > 0: sys.stderr.write('\ndim: %d' % c)

            if c < min_c:
                Y = numpy.array(ConditionalsBinary.state_space(c), dtype=numpy.int8)
                pm = numpy.exp(ConditionalsBinary._rvslpmf_all(A=A[:c, :c], Y=Y)[1])
            else:
                Y = numpy.empty(shape=(n, c), dtype=numpy.int8)
                U = numpy.random.random(size=(n, c))

                # sample array of random binary vectors
                for k in xrange(n):

                    for i in xrange(c):

                        # compute the probability that Y(k,i) is one                    
                        x = A[i, i]
                        for j in xrange(i): x += A[i, j] * Y[k, j]

                        # generate the entry Y(k,i)
                        Y[k, i] = U[k, i] < 1.0 / (1.0 + exp(-x))

            # filter components with high association for sparse regression
            S = numpy.append((abs(corr[c, :c]) > delta).nonzero(), c)
            s = S.shape[0] - 1

            # initialize b with independent parameter
            a = numpy.zeros(s + 1, dtype=numpy.float64)
            a[s] = ConditionalsBinary.ilink(M[c, c])
            A[c, S] = a

            # set target moment vector and independent moment vector
            tM, tI = M[c, S], I[c, S]

            # Newton-Raphson iteration
            for phi in Q:

                for nr in xrange(ConditionalsBinary.MAX_ITERATIONS):

                    if verbose > 1: sys.stderr.write('phi: %.3f, nr: %d, a: %s' % (phi, nr, repr(A)))
                    a_before = a.copy()

                    # compute f and J 
                    f = numpy.zeros(s + 1, dtype=numpy.float64)
                    J = numpy.zeros((s + 1, s + 1), dtype=numpy.float64)

                    # loop over all binary vectors
                    for k in xrange(Y.shape[0]):

                        x = a[s]
                        for i in xrange(s): x += a[i] * Y[k, S[i]]

                        prob = ConditionalsBinary.link(x) # link
                        prob = min(max(prob, 1e-8), 1.0 - 1e-8)

                        x = prob / float(n)
                        if c < min_c: x *= n * pm[k]
                        for i in xrange(s):
                            f[i] = f[i] + x * Y[k, S[i]]
                        f[s] = f[s] + x

                        x = ConditionalsBinary.dlink(x) / float(n) # dlink
                        if c < min_c: x *= n * pm[k]
                        for i in xrange(s):
                            for j in xrange(s):
                                J[i, j] += x * Y[k, S[i]] * Y[k, S[j]]
                            J[s, i] += x * Y[k, S[i]]
                            J[i, s] += x * Y[k, S[i]]
                        J[s, s] += x

                    # subtract non-random parts
                    f -= (phi * tM + (1.0 - phi) * tI)

                    # Newton update
                    try:
                        a = scipy.linalg.solve(J, numpy.dot(J, a) - f, sym_pos=True)
                    except numpy.linalg.linalg.LinAlgError:
                        sys.stderr.write('numerical error. adding 1e-8 on main diagonal.')
                        a = scipy.linalg.solve(J + numpy.eye(s + 1) * 1e-8, numpy.dot(J, a) - f, sym_pos=False)

                    # check for absolute sums in A
                    entry_sum = max(a[a > 0].sum(), -a[a < 0].sum())
                    if entry_sum > ConditionalsBinary.MAX_ENTRY_SUM * (0.25 * s + 1):
                        if verbose > 1: sys.stderr.write('stopped. a exceeding %.1f\n' % entry_sum)
                        nr = None
                        break

                    # check for convergence
                    if numpy.allclose(a, a_before, rtol=0, atol=ConditionalsBinary.PRECISION):
                        if verbose > 1: sys.stderr.write('a converged.\n')
                        A[c, S] = a
                        break

                if nr is None or phi == 1.0:
                    if verbose: sys.stderr.write('phi: %.3f ' % phi)
                    break


        if verbose: sys.stderr.write('\nlogistic conditionals family successfully constructed from moments.\n\n')

        return cls(A)

    @classmethod
    def test_properties(cls, d, n=1e4, rho=0.8, ncpus=1):
        """
            Tests functionality of the quadratic linear family class.
            \param d dimension
            \param n number of samples
            \param phi dependency level in [0,1]
            \param ncpus number of cpus 
        """

        mean, corr = base.moments2corr(base.random_moments(d, rho=rho))
        print 'given marginals '.ljust(100, '*')
        base.print_moments(mean, corr)

        generator = ConditionalsBinary.from_moments(mean, corr)
        print generator.name + ':'
        print generator

        print 'exact '.ljust(100, '*')
        base.print_moments(generator.exact_marginals(ncpus))

        print ('simulation (n = %d) ' % n).ljust(100, '*')
        base.print_moments(generator.rvs_marginals(n, ncpus))

    @classmethod
    def link(cls, x):
        """ Logistic function 1/(1+exp(x)) \return logistic function """
        return 1.0 / (1.0 + numpy.exp(-x))

    @classmethod
    def dlink(cls, x):
        """ Derivative of logistic function 1/(1+exp(x)) \return derivative of logistic function """
        p = ConditionalsBinary.link(x)
        return p * (1 - p)

    @classmethod
    def ilink(cls, p):
        """ Inverse of logistic function exp(1/(1-p)) \return logit function """
        return numpy.log(p / (1.0 - p))
