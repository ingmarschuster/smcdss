#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with quadratic form. \namespace binary.qu_linear"""

import numpy
import scipy.linalg
import binary.base as base
import binary.wrapper as wrapper


class QuLinearBinary(base.BaseBinary):
    """ Binary parametric family with quadratic linear form. """

    def __init__(self, a, Beta):
        """
            Constructor.
            \param a probability of zero
            \param Beta matrix of coefficients
        """
        super(QuLinearBinary, self).__init__(self, d=Beta.shape[0], name='quadratic linear binary', long_name=__doc__)

        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg', 'binary.qu_linear')

        # add dependent functions
        self.pp_depfuncs += ('_pmf',)

        self.py_wrapper = wrapper.qu_linear()

        #scipy.linalg.cholesky(Beta)

        # normalize parameters
        z = 2 ** self.d * (a + 0.25 * (Beta.sum() + Beta.trace()))
        self.a = a / z
        self.Beta = Beta / z

        ## probability of first entry
        self.p = self.mean[0]

    def __str__(self):
        return 'd: %d, a:%.4f, Beta:\n%s' % (self.d, self.a, repr(self.Beta))

    def pmf_nonnegative(self, Y):
        """ 
            Probability mass function.
            \param Y binary vector
            \return probabilities
        """
        return numpy.maximum(self.pmf_quadratic(Y), 0.0)

    def pmf_quadratic(self, Y):
        """ 
            Probability mass function.
            \param Y binary vector
            \return probabilities
        """

        if Y.ndim == 1: Y = Y[numpy.newaxis, :]
        P = numpy.empty(Y.shape[0])

        for k in xrange(Y.shape[0]):
            v = float(numpy.dot(numpy.dot(Y[k], self.Beta), Y[k].T))
            P[k] = (self.a + v)

        if Y.shape[0] == 1: return P[0]
        else: return P

    @classmethod
    def _rvslpmf_all(cls, Beta, p, U=None, Y=None):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """

        if U is not None:
            size, d = U.shape
            Y = numpy.empty(U.shape, dtype=bool)

        if Y is not None:
            size, d = Y.shape

        L = numpy.empty(size, dtype=float)

        for k in xrange(size):

            if U is not None:
                Y[k, 0] = U[k, 0] < p

            # initialize
            if Y[k, 0]:
                prob_previous = p
            else:
                prob_previous = 1.0 - p

            L[k] = numpy.log(prob_previous)

            for m in xrange(1, d):

                # compute k - marginal with gamma(k) = 1
                tmp = 0.0
                for i in xrange(d):
                    tmp = tmp + (2 * Y[k, i] * (i <= m - 1) + (i > m - 1)) * Beta[i, m]

                tmp = tmp * 2 ** (d - (m + 2))
                prob = prob_previous / 2.0 + tmp

                # compute conditional probability
                prob_cond = min(max(prob / prob_previous, 0.0), 1.0)

                # draw m th entry
                if U is not None:
                    Y[k, m] = U[k, m] < prob_cond

                # recompute k - marginal after draw
                if Y[k, m]:
                    prob_previous = prob
                    if prob_cond == 0.0:
                        L[k] = -numpy.inf
                    else:
                        L[k] += numpy.log(prob_cond)
                else:
                    prob_previous = prob_previous - prob
                    if prob_cond == 1.0:
                        L[k] = -numpy.inf
                    else:
                        L[k] += numpy.log(1.0 - prob_cond)
        return Y, L

    @classmethod
    def random(cls, d):
        """
            Construct a random linear model for testing.
            \param cls class
            \param d dimension
        """
        Beta = numpy.random.standard_cauchy(size=(d, d))
        Beta = numpy.dot(Beta.T, Beta)
        return cls(0.0, Beta)

    @classmethod
    def from_moments(cls, mean, corr):
        """
            Constructs a linear model for given moments. Warning: This method
            might produce parameters that are infeasible and yield an improper
            distribution.
            \param mean mean
            \param corr correlation matrix
        """
        d = mean.shape[0]

        # generate data independent Z-matrix
        Z = QuLinearBinary.generate_Z(d)

        # compute cross-moment matrix
        M = base.corr2moments(mean, corr)

        # convert adjusted moments to vector
        s = m2v(2 * M - numpy.diag(numpy.diag(M)))

        # add normalization constant
        s = numpy.array(list(s) + [1.0])

        # solve moment equation
        x = scipy.linalg.solve(Z, s)

        a, Beta = x[-1], v2m(x[:-1])

        return cls(a, Beta)

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
    def generate_Z(cls, d):
        """
            Generates a design matrix for the method of moments.
            \param dimension
            \return design matrix
        """
        dd = d * (d + 1) / 2 + 1;
        Z = numpy.ones((dd, dd))
        for i in range(d):
            for j in range(i, d):
                for k in range(d):
                    for l in range(k, d):
                        Z[tau(i, j), tau(k, l)] = 2 * (
        (1 + (k == i or k == j)) * (1 + (l == i or l == j)) * (l != k) + (1 + (k == i or k == j)) * (l == k))

        Z[:-1, :-1] /= 4.0
        Z[-1, -1] = 2.0
        return Z

    @classmethod
    def test_properties(cls, d, n=1e4, phi=0.8, ncpus=1):
        """
            Tests functionality of the quadratic linear family class.
            \param d dimension
            \param n number of samples
            \param phi dependency level in [0,1]
            \param ncpus number of cpus 
        """

        mean, corr = base.moments2corr(base.random_moments(d, phi=phi))
        print 'given marginals '.ljust(100, '*')
        base.print_moments(mean, corr)

        generator = QuLinearBinary.from_moments(mean, corr)
        print generator.name + ':'
        print generator

        print 'formula '.ljust(100, '*')
        base.print_moments(generator.mean, generator.corr)

        print 'exact \pi conditionals in [0,1] '.ljust(100, '*')
        base.print_moments(generator.exact_marginals(ncpus))

        print ('simulation \pi conditionals in [0,1] (n = %d) ' % n).ljust(100, '*')
        base.print_moments(generator.rvs_marginals(n, ncpus))

        print 'exact \pi non-negative '.ljust(100, '*')
        X = generator.state_space(d)
        mean, corr = base.sample2corr(X, generator.pmf_nonnegative(X))
        base.print_moments(mean, corr)

    def _getMean(self):
        """ Get expected value of instance. \return mean """
        return 0.5 + self.Beta.sum(axis=0) * 2 ** (self.d - 2)

    def getCov(self):
        """ Get covariance matrix. \return covariance matrix """
        A = numpy.outer(numpy.ones(self.d), self.Beta.sum(axis=0))
        S = 0.25 + (A + A.T + self.Beta) * 2 ** (self.d - 3)
        mean = self.mean
        for i in range(self.d): S[i, i] = mean[i]
        return S - numpy.outer(mean, mean)

    def getCorr(self):
        """ Get correlation matrix. \return correlation matrix """
        cov = self.cov
        var = cov.diagonal()
        return cov / numpy.sqrt(numpy.outer(var, var))

    cov = property(fget=getCov, doc="covariance matrix")
    corr = property(fget=getCorr, doc="correlation matrix")


def tau(i, j):
    """
        Maps the indices of a symmetric matrix onto the indices of a vector.
        \param i matrix index
        \param j matrix index
        \return vector index
    """
    return j * (j + 1) / 2 + i

def m2v(A):
    """
        Turns a symmetric matrix into a vector.
        \param matrix
        \return vector
    """
    d = A.shape[0]
    a = numpy.zeros(d * (d + 1) / 2)
    for i in range(d):
        for j in range(i, d):
            a[tau(i, j)] = A[i, j]
    return a

def v2m(a):
    """
        Turns a vector into a symmetric matrix.
        \param vector
        \return matrix
    """
    d = int((numpy.sqrt(1 + 8 * a.shape[0]) - 1) / 2)
    A = numpy.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            A[i, j] = a[tau(i, j)]
            A[j, i] = A[i, j]
    return A
