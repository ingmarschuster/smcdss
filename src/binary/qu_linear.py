#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with quadratic form. """

"""
\namespace binary.qu_linear
$Author: christian.a.schafer@gmail.com $
$Rev: 122 $
$Date: 2011-04-12 19:22:11 +0200 (Di, 12 Apr 2011) $
"""

import numpy
import scipy.linalg
import binary.base
import binary.wrapper


class QuLinearBinary(binary.base.BaseBinary):
    """ Binary parametric family with quadratic linear form. """

    def __init__(self, a, Beta):
        """
            Constructor.
            \param a probability of zero vector
            \param Beta matrix of coefficients
        """
        binary.base.BaseBinary.__init__(self, d=Beta.shape[0], name='quadratic linear binary', long_name=__doc__)

        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg', 'binary.qu_linear')

        # add dependent functions
        self.pp_depfuncs += ('_pmf',)

        self.py_wrapper = binary.wrapper.qu_linear()

        #scipy.linalg.cholesky(Beta)

        # normalize parameters
        z = 2 ** self.d * (a + 0.25 * (Beta.sum() + Beta.trace()))
        self.a = a / z
        self.Beta = Beta / z

        ## probability of first entry
        self.p = self.mean[0]

    def __str__(self):
        return 'd: %d, a:%.4f, Beta:\n%s' % (self.d, self.a, repr(self.Beta))

    def pmf(self, Y):
        return QuLinearBinary._pmf(Y, self.Beta, self.a)

    @classmethod
    def _pmf(cls, Y, Beta, a):
        """ 
            Probability mass function.
            \param Y binary vector
            \return probabilities
        """

        if Y.ndim == 1: Y = Y[numpy.newaxis, :]
        L = numpy.empty(Y.shape[0])

        for k in xrange(Y.shape[0]):
            v = float(numpy.dot(numpy.dot(Y[k], Beta), Y[k].T))
            L[k] = (a + v)

        if Y.shape[0] == 1: return L[0]
        else: return L

    @classmethod
    def _lpmf(cls, Y, Beta, a):
        """ 
            Log-probability mass function.
            \param Y binary vector
            \param param parameters
            \return log-probabilities
        """
        L = QuLinearBinary._pmf(Y, Beta, a)
        L[L <= 0] = -numpy.inf * (L <= 0).shape[0]
        L[L > 0] = numpy.log(L[L > 0])
        return L

    def pmf_reject(self, Y):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        Beta = self.Beta
        p = self.p

        size, d = Y.shape
        P = numpy.zeros(size, dtype=float)

        for k in xrange(size):

            if Y[k, 0]:
                P[k] = p
            else:
                P[k] = (1.0 - p)

            # initialize
            previous_p = P[k]

            for m in range(1, d):

                # compute k - marginal with gamma(k) = 1
                tmp = 0
                for i in xrange(d):
                    tmp = tmp + (2 * Y[k, i] * (i <= m - 1) + (i > m - 1)) * Beta[i, m]

                tmp = tmp * 2 ** (d - (m + 2))
                prob = previous_p / 2.0 + tmp

                # compute conditional probability
                p_cond = prob / previous_p

                if p_cond < 0 or p_cond > 1:
                    P[k] = 0.0
                    break

                # recompute k - marginal after draw
                if Y[k, m]:
                    P[k] *= p_cond
                    previous_p = prob
                else:
                    previous_p = previous_p - prob
                    P[k] *= (1.0 - p_cond)
        return P

    @classmethod
    def _rvs(cls, U, Beta, p):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        size, d = U.shape
        Y = numpy.zeros(U.shape, dtype=bool)

        for k in xrange(size):

            while True:
                reject = False
                Y[k, 0] = numpy.random.random() < p

                # initialize
                if Y[k, 0]: previous_p = p
                else: previous_p = 1 - p

                for m in range(1, d):

                    # compute k - marginal with gamma(k) = 1
                    tmp = 0
                    for i in xrange(d):
                        tmp = tmp + (2 * Y[k, i] * (i <= m - 1) + (i > m - 1)) * Beta[i, m]

                    tmp = tmp * 2 ** (d - (m + 2))
                    prob = previous_p / 2.0 + tmp

                    # compute conditional probability
                    p_cond = prob / previous_p;

                    if p_cond < 0 or p_cond > 1:
                        reject = True
                        break

                    # draw k th entry
                    Y[k, m] = numpy.random.random() < p_cond

                    # recompute k - marginal after draw
                    if Y[k, m]: previous_p = prob
                    else: previous_p = previous_p - prob

                # accept
                if not reject: break

        return Y

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
        M = binary.base.corr2moments(mean, corr)

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


def main():
    pass

if __name__ == "__main__":
    main()
