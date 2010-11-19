#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-10-29 13:41:14 +0200 (ven., 29 oct. 2010) $
    $Revision: 29 $
'''

from numpy import *
from auxpy.data import *
from binary import Binary
from scipy.linalg import cholesky, LinAlgError

class LinearBinary(Binary):
    '''
        A multivariate Bernoulli with additive probability mass function.
    '''

    def __init__(self, a, Beta):
        '''
            Constructor.
            @param a probability of zero vector
            @param Beta matrix of coefficients
        '''
        Binary.__init__(self, name='linear-binary', longname='A multivariate Bernoulli with additive probability mass function.')

        ## probability of zero vector
        self.a = a
        ## matrix of coefficients 
        self.Beta = Beta
        ## normalization constant
        self.c = 2 ** self.d * (self.a + 0.25 * (self.Beta.sum() + diag(self.Beta).sum()))

        try:
            cholesky(self.Beta)
        except LinAlgError:
            print 'Warning: The linear model might not be a proper distribution.'

    @classmethod
    def random(cls, d):
        '''
            Construct a random linear model for testing.
            @param cls class
            @param d dimension
        '''
        Beta = random.normal(scale=5.0, size=(d, d))
        Beta = dot(Beta.T, Beta)
        return cls(0.0, Beta)

    @classmethod
    def from_moments(cls, p, R):
        '''
            Constructs a linear model for given moments. Warning: This method might produce
            parameters that are infeasible and yield an improper distribution. 
            @param cls class 
            @param d dimension
        '''
        result = calc_Beta(p, R)
        return cls(result[0], result[1])

    @classmethod
    def from_data(cls, sample):
        '''
            Constructs a linear model from data. Warning: This method might produce
            parameters that are infeasible and yield an improper distribution.
            @param cls class 
            @param d dimension
        '''
        return cls.from_moments(sample.mean, sample.cor)

    def _pmf(self, gamma):
        '''
            Probability mass function.
            @param gamma: binary vector
        '''
        v = float(dot(dot(gamma[newaxis, :], self.Beta), gamma[:, newaxis]))
        return (self.a + v) / self.c

    def _rvs(self):
        '''
            Samples from the model.
            @return random variable
        '''
        gamma = zeros(self.d, dtype=bool)
        gamma[0] = self.p[0] > random.random()

        # initialize
        if gamma[0]: previous_p = self.p[0]
        else: previous_p = 1 - self.p[0]

        for k in range(1, self.d):

            # compute k - marginal with gamma(k) = 1
            tmp = 0
            for i in range(self.d):
                tmp = tmp + (2 * gamma[i] * (i <= k - 1) + (i > k - 1)) * self.Beta[i, k]

            tmp = tmp * 2 ** (self.d - (k + 2)) / self.c
            p = previous_p / 2.0 + tmp

            # compute conditional probability
            p_cond = p / previous_p;

            # draw k th entry
            gamma[k] = p_cond > random.random()

            # recompute k - marginal after draw
            if gamma[k]: previous_p = p
            else: previous_p = previous_p - p

        return gamma

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.Beta.shape[0]

    def getP(self):
        '''
            Get mean.
            @return dimension 
        '''
        if not hasattr(self, '__p'):
            self.__p = 0.5 + self.Beta.sum(axis=0) * 2 ** (self.d - 2) / self.c
        return self.__p

    def getR(self):
        '''
            Get correlation matrix.
            @return dimension 
        '''
        if not hasattr(self, '__R'):
            A = dot(ones(self.d)[:, newaxis], self.Beta.sum(axis=0)[newaxis, :])
            S = 0.25 + (A + A.T + self.Beta) * 2 ** (self.d - 3) / self.c
            for i in range(self.d): S[i, i] = p[i]
            cov = S - dot(self.p[:, newaxis], self.p[newaxis, :])
            var = diag(cov)
            self.__R = cov / sqrt(dot(var[:, newaxis], var[newaxis, :]))
        return self.__R

    d = property(fget=getD, doc="dimension")
    p = property(fget=getP, doc="mean")
    R = property(fget=getR, doc="correlation")


def tau(i,j):
    '''
        Maps the indices of a symmetric matrix onto the indices of a vector.
        @param i matrix index
        @param j matrix index
        @return vector index
    '''
    return j * (j + 1) / 2 + i


def m2v(A):
    '''
        Turns a symmetric matrix into a vector.
        @param matrix
        @return vector
    '''
    d = A.shape[0]
    a = zeros(d * (d + 1) / 2)
    for i in range(d):
        for j in range(i, d):
            a[tau(i, j)] = A[i, j]
    return a

def v2m(a):
    '''
        Turns a vector into a symmetric matrix.
        @param vector
        @return matrix
    '''
    d = a.shape[0]
    d = int((sqrt(1 + 8 * d) - 1) / 2)
    A = zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            A[i, j] = a[tau(i, j)]
            A[j, i] = A[i, j]
    return A

def generate_Z(d):
    '''
        Generates a design matrix for the method of moments.
        @param dimension
        @return design matrix
    '''
    dd = d * (d + 1) / 2 + 1;
    Z = ones((dd, dd))
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
    '''
        Computes the coefficients that yield a linear distribution with mean p and correlation R.
        @param p mean
        @param R correlation
        @return a, Beta
    '''
    d = p.shape[0]

    # generate data independent Z-matrix
    Z = generate_Z(d)

    # compute second raw moment
    var = p * (1 - p)
    S = R * sqrt(dot(var[:, newaxis], var[newaxis, :])) + dot(p[:, newaxis], p[newaxis, :])
    S = 2 * S - diag(diag(S))

    # convert moments to vector
    s = m2v(S)

    # add normalization constant
    s = array(list(s) + [1.0])

    # solve moment equation
    x = linalg.solve(Z, s)

    return x[-1], v2m(x[:-1])
