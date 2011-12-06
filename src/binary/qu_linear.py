#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with quadratic form. """

"""
\namespace binary.qu_linear
$Author$
$Rev$
$Date$
"""

import numpy
import scipy.linalg
import base

def _pmf(Y, param):
    """ 
        Probability mass function.
        \param Y binary vector
        \param param parameters
        \return probabilities
    """
    Beta, a, c = param['Beta'], param['a'], param['c']
    
    if Y.ndim == 1: Y = Y[numpy.newaxis, :]
    L = numpy.empty(Y.shape[0])

    for k in xrange(Y.shape[0]):
        v = float(numpy.dot(numpy.dot(Y[k], Beta), Y[k].T))
        L[k] = (a + v) / c

    if Y.shape[0] == 1: return L[0]
    else: return L

def _lpmf(Y, param):
    """ 
        Log-probability mass function.
        \param Y binary vector
        \param param parameters
        \return log-probabilities
    """
    L = _pmf(Y, param)
    L[L <= 0] = 1.0
    return numpy.log(L)

def _rvs(U, param):
    """ 
        Generates a random variable.
        \param U uniform variables
        \param param parameters
        \return binary variables
    """
    Beta, c, mean = param['Beta'], param['c'], param['mean']

    d = U.shape[1]
    Y = numpy.zeros((U.shape[0], U.shape[1]), dtype=bool)
    
    for k in xrange(U.shape[0]):

        Y[k, 0] = mean[0] > U[k, 0]
    
        # initialize
        if Y[k, 0]: previous_p = mean[0]
        else: previous_p = 1 - mean[0]
    
        for m in range(1, d):
    
            # compute k - marginal with gamma(k) = 1
            tmp = 0
            for i in xrange(d):
                tmp = tmp + (2 * Y[k, i] * (i <= m - 1) + (i > m - 1)) * Beta[i, m]
    
            tmp = tmp * 2 ** (d - (m + 2)) / c
            prob = previous_p / 2.0 + tmp
    
            # compute conditional probability
            p_cond = prob / previous_p;
    
            # draw k th entry
            Y[k, m] = p_cond > numpy.random.random()
    
            # recompute k - marginal after draw
            if Y[k, m]: previous_p = prob
            else: previous_p = previous_p - prob

    return Y

def _rvslpmf(U, param):
    """ 
        Generates a random variable and computes its probability.
        \param U uniform variables
        \param param parameters
        \return binary variables, log-probabilities
    """
    Y = _rvs(U, param)
    return Y, _lpmf(Y, param)


class QuLinearBinary(base.BaseBinary):
    """ Binary parametric family with quadratic form. """

    def __init__(self, a, Beta):
        """
            Constructor.
            \param a probability of zero vector
            \param Beta matrix of coefficients
        """
        base.BaseBinary.__init__(self, pp_modules=('numpy',),
                                 pp_depfuncs={'lpmf':_lpmf, 'rvs': _rvs, 'rvslpmf':_rvslpmf, 'pmf':_pmf},
                                 name='quadratic binary', long_name=__doc__)

        self.param.update({'Beta':Beta, 'a':a})

        # normalization constant
        c = 2 ** self.d * (a + 0.25 * (Beta.sum() + numpy.diag(Beta).sum()))
        self.param.update({'c':c})
        
        # mean
        self.param.update({'mean':self.mean})

        #try:
        #    cholesky(self.Beta)
        #except LinAlgError:
        #    print 'Warning: The linear model might not be a proper distribution.'

    @classmethod
    def random(cls, d):
        """
            Construct a random linear model for testing.
            \param cls class
            \param d dimension
        """
        Beta = numpy.random.normal(scale=5.0, size=(d, d))
        Beta = numpy.dot(Beta.T, Beta)
        return cls(0.0, Beta)

    @classmethod
    def from_moments(cls, p, R):
        """
            Constructs a linear model for given moments. Warning: This method
            might produce parameters that are infeasible and yield an improper
            distribution.
            \param cls class 
            \param d dimension
        """
        result = calc_Beta(p, R)
        return cls(result[0], result[1])

    @classmethod
    def from_data(cls, sample):
        """
            Constructs a linear model from data. Warning: This method might
            produce parameters that are infeasible and yield an improper
            distribution.
            \param cls class 
            \param d dimension
        """
        return cls.from_moments(sample.mean, sample.cor)

    def _getD(self):
        """ Get dimension of instance. \return dimension """
        return self.param['Beta'].shape[0]

    def _getMean(self):
        """ Get expected value of instance. \return mean """
        return 0.5 + self.param['Beta'].sum(axis=0) * 2 ** (self.d - 2) / self.param['c']

    def getR(self):
        """ Get correlation matrix. \return correlation matrix """
        if not hasattr(self, '__R'):
            A = numpy.dot(numpy.ones(self.d)[:, numpy.newaxis], self.Beta.sum(axis=0)[numpy.newaxis, :])
            S = 0.25 + (A + A.T + self.Beta) * 2 ** (self.d - 3) / self.c
            for i in range(self.d): S[i, i] = self.p[i]
            cov = S - numpy.dot(self.p[:, numpy.newaxis], self.p[numpy.newaxis, :])
            var = numpy.diag(cov)
            self.__R = cov / numpy.sqrt(numpy.dot(var[:, numpy.newaxis], var[numpy.newaxis, :]))
        return self.__R

    R = property(fget=getR, doc="correlation")


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
        for j in range(i + 1):
            a[tau(i, j)] = A[i, j]
    return a

def v2m(a):
    """
        Turns a vector into a symmetric matrix.
        \param vector
        \return matrix
    """
    d = a.shape[0]
    d = int((numpy.sqrt(1 + 8 * d) - 1) / 2)
    A = numpy.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            A[i, j] = a[tau(i, j)]
            A[j, i] = A[i, j]
    return A

def generate_Z(d):
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

def calc_Beta(p, R):
    """
        Computes the coefficients that yield a linear distribution with mean p and correlation R.
        \param p mean
        \param R correlation
        \return a, Beta
    """
    d = p.shape[0]

    # generate data independent Z-matrix
    Z = generate_Z(d)

    # compute second raw moment
    var = p * (1 - p)
    S = R * numpy.sqrt(numpy.dot(var[:, numpy.newaxis], var[numpy.newaxis, :])) + numpy.dot(p[:, numpy.newaxis], p[numpy.newaxis, :])
    S = 2 * S - numpy.diag(numpy.diag(S))

    # convert moments to vector
    s = m2v(S)

    # add normalization constant
    s = numpy.array(list(s) + [1.0])

    # solve moment equation
    x = scipy.linalg.solve(Z, s)

    return x[-1], v2m(x[:-1])


def main():
    h = QuLinearBinary.random(3)
    print h.mean
    print h.rvstest(5000)

if __name__ == "__main__":
    main()
