#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with logistic conditionals. \namespace binary.conditonals"""

import binary.base as base
import binary.product as product
import binary.wrapper as wrapper
import numpy
import scipy.linalg
import sys
cimport numpy

cdef extern from "math.h":
    double exp(double)
    double log(double)


class ConditionalsBinary(product.ProductBinary):
    """ Binary parametric family with glm conditionals. """

    PRECISION = base.BaseBinary.PRECISION
    MAX_ENTRY_SUM = numpy.finfo(float).maxexp * log(2)

    name = 'generic conditionals family'

    def __init__(self, A, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param A Lower triangular matrix holding regression coefficients
            \param name name
            \param long_name long name
        """

        p = self.link(numpy.diagonal(A))

        # call super constructor
        super(ConditionalsBinary, self).__init__(p=p, name=name, long_name=long_name)

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
            \param A parameters
            \param Y array of binary vectors
            \return array of binary vectors, log-likelihood
        """
        cdef Py_ssize_t d = A.shape[0]
        cdef Py_ssize_t k, i, j, size
        cdef double ax, cprob

        if U is not None:
            size = U.shape[0]
            Y = numpy.empty((size, d), dtype=numpy.int8)

        if Y is not None:
            size = Y.shape[0]

        cdef numpy.ndarray[dtype = numpy.float64_t, ndim = 1] L = numpy.ones(size, dtype=numpy.float64)

        for k in xrange(size):

            for i in xrange(d):
                # Compute log conditional probability that Y(i) is one
                ax = A[i, i]
                for j in xrange(i): ax += A[i, j] * Y[k, j]
                cprob = cls.link(ax)

                # Generate the ith entry
                if U is not None:
                    Y[k, i] = U[k, i] < cprob

                # Add to log conditional probability
                if Y[k, i]: L[k] *= cprob
                else: L[k] *= (1.0 - cprob)

        return numpy.array(Y, dtype=bool), numpy.log(L)

    @classmethod
    def independent(cls, p):
        """
            Constructs a conditionals family with independent components.
            \param cls instance
            \param p mean
            \return conditionals model
        """
        A = numpy.diag(cls.ilink(p))
        return cls(A)

    @classmethod
    def uniform(cls, d):
        """ 
            Constructs a uniform conditionals family.
            \param cls instance
            \param d dimension
            \return conditionals family
        """
        A = numpy.zeros((d, d))
        return cls(A)

    @classmethod
    def random(cls, d, dep=3.0):
        """ 
            Constructs a random conditionals family.
            \param cls instance
            \param d dimension
            \param dep strength of dependencies [0,inf)
            \return conditionals family
        """
        cls = ConditionalsBinary.independent(p=numpy.random.random(d))
        A = numpy.random.normal(scale=dep, size=(d, d))
        A *= numpy.dot(A, A)
        for i in xrange(d): A[i, i] = cls.A[i, i]
        cls.A = A
        return cls

    @classmethod
    def from_moments(cls, mean, corr, n=1e6, q=25.0, delta=0.005, verbose=0):
        """ 
            Constructs a conditionals family from given mean and correlation.
            \param mean mean
            \param corr correlation
            \param n number of samples for Monte Carlo estimation
            \param q number of intermediate steps in Newton-Raphson procedure
            \param delta minimum absolute value of correlation coefficients
            \return conditionals family
        """

        ## dimension of binary family
        cdef Py_ssize_t d = mean.shape[0]

        ## dimension of the current regression
        cdef Py_ssize_t c

        ## dimension of the sparse regression
        cdef Py_ssize_t s

        ## iterators
        cdef Py_ssize_t k, i, j

        ## minimum dimension for Monte Carlo estimates
        cdef Py_ssize_t min_c = int(numpy.log2(n))

        ## floating point variables
        cdef double x, ax, link, dlink, high, low

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

        ## index vector for sparse regression
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
        A[0, 0] = cls.ilink(M[0, 0])

        # loop over dimensions
        for c in xrange(1, d):
            if verbose > 0: sys.stderr.write('\ndim: %d' % c)

            if c < min_c:
                Y = numpy.array(base.state_space(c), dtype=numpy.int8)
                pm = numpy.exp(cls._rvslpmf_all(A=A[:c, :c], Y=Y)[1])
            else:
                Y = numpy.empty(shape=(n, c), dtype=numpy.int8)
                U = numpy.random.random(size=(n, c))

                # sample array of random binary vectors
                for k in xrange(n):

                    for i in xrange(c):

                        # compute the probability that Y(k,i) is one                    
                        ax = A[i, i]
                        for j in xrange(i): ax += A[i, j] * Y[k, j]

                        # generate the entry Y(k,i)
                        Y[k, i] = U[k, i] < cls.link(ax)

            # filter components with high association for sparse regression
            S = numpy.append((abs(corr[c, :c]) > delta).nonzero(), c)
            s = S.shape[0] - 1

            # initialize b with independent parameter
            a = numpy.zeros(s + 1, dtype=numpy.float64)
            a[s] = cls.ilink(M[c, c])
            A[c, S] = a

            # set target moment vector and independent moment vector
            tM, tI = M[c, S], I[c, S]

            # Newton-Raphson iteration
            for phi in Q:

                for nr in xrange(cls.MAX_ITERATIONS):

                    if verbose > 1: sys.stderr.write('\nphi: %.3f, nr: %d, a: %s' % (phi, nr, repr(a)))
                    a_before = a.copy()

                    # compute f and J 
                    f = numpy.zeros(s + 1, dtype=numpy.float64)
                    J = numpy.zeros((s + 1, s + 1), dtype=numpy.float64)

                    # loop over all binary vectors
                    for k in xrange(Y.shape[0]):

                        # compute sum
                        ax = a[s]
                        for i in xrange(s): ax += a[i] * Y[k, S[i]]

                        # link
                        link = cls.link(ax)
                        link = min(max(link, 1e-8), 1.0 - 1e-8)
                        if c < min_c:
                            x = link * pm[k]
                        else:
                            x = link / float(n)
                        # update f
                        for i in xrange(s):
                            f[i] = f[i] + x * Y[k, S[i]]
                        f[s] = f[s] + x

                        # derivative of link
                        dlink = cls.dlink(ax)
                        if c < min_c:
                            x = dlink * pm[k]
                        else:
                            x = dlink / float(n)
                        # update J
                        for i in xrange(s):
                            for j in xrange(s):
                                J[i, j] += x * Y[k, S[i]] * Y[k, S[j]]
                            J[s, i] += x * Y[k, S[i]]
                            J[i, s] += x * Y[k, S[i]]
                        J[s, s] += x

                    # subtract non-random parts
                    m = phi * tM + (1.0 - phi) * tI

                    # Newton update
                    try:
                        a -= scipy.linalg.solve(J, f - m, sym_pos=True)
                    except numpy.linalg.linalg.LinAlgError:
                        if verbose > 1: sys.stderr.write('numerical error. adding 1e-8 on main diagonal.\n')
                        a -= scipy.linalg.solve(J + numpy.eye(s + 1) * 1e-8, f - m, sym_pos=False)

                    # check for absolute sums in A
                    entry_sum = max(a[a > 0].sum(), -a[a < 0].sum())
                    if entry_sum > cls.MAX_ENTRY_SUM * (0.1 * s + 1):
                        if verbose > 1: sys.stderr.write('stopped. a exceeding %.1f\n' % entry_sum)
                        nr = None
                        break

                    # check for convergence
                    if numpy.allclose(a, a_before, rtol=0, atol=cls.PRECISION):
                        if verbose > 1: sys.stderr.write('a converged.\n')
                        A[c, S] = a
                        break

                if nr is None or phi == 1.0:
                    if verbose: sys.stderr.write('phi: %.3f ' % phi)
                    break

        if verbose: sys.stderr.write('\nconditionals family successfully constructed from moments.\n\n')

        return cls(A)

    @classmethod
    def link(cls, x):
        pass

    @classmethod
    def dlink(cls, x):
        pass

    @classmethod
    def ilink(cls, p):
        pass
