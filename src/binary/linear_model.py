#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import numpy
import utils
import binary

from binary import Binary
import scipy.linalg

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
        self.f_lpmf = None
        self.f_rvs = None
        self.f_rvslpmf = None
        self.param = dict(Beta=Beta, a=a)

        ## normalization constant
        c = 2 ** self.d * (a + 0.25 * (Beta.sum() + numpy.diag(Beta).sum()))
        self.param.update({'c':c})

        #try:
        #    cholesky(self.Beta)
        #except LinAlgError:
        #    print 'Warning: The linear model might not be a proper distribution.'

    @classmethod
    def random(cls, d):
        '''
            Construct a random linear model for testing.
            @param cls class
            @param d dimension
        '''
        Beta = numpy.random.normal(scale=5.0, size=(d, d))
        Beta = numpy.dot(Beta.T, Beta)
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

    def pmf(self, gamma, job_server=None):
        ''' Probability mass function.
            @param gamma binary vector
        '''
        return _pmf(gamma, self.param)

    def _rvs(self):
        '''
            Samples from the model.
            @return random variable
        '''
        gamma = numpy.zeros(self.d, dtype=bool)
        gamma[0] = self.p[0] > numpy.random.random()

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
            gamma[k] = p_cond > numpy.random.random()

            # recompute k - marginal after draw
            if gamma[k]: previous_p = p
            else: previous_p = previous_p - p

        return gamma

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.param['Beta'].shape[0]

    def getP(self):
        '''
            Get mean.
            @return dimension 
        '''
        if not hasattr(self, '__p'):
            self.__p = 0.5 + self.param['Beta'].sum(axis=0) * 2 ** (self.d - 2) / self.param['c']
        return self.__p

    def getR(self):
        '''
            Get correlation matrix.
            @return dimension 
        '''
        if not hasattr(self, '__R'):
            A = numpy.dot(numpy.ones(self.d)[:, numpy.newaxis], self.Beta.sum(axis=0)[numpy.newaxis, :])
            S = 0.25 + (A + A.T + self.Beta) * 2 ** (self.d - 3) / self.c
            for i in range(self.d): S[i, i] = self.p[i]
            cov = S - numpy.dot(self.p[:, numpy.newaxis], self.p[numpy.newaxis, :])
            var = numpy.diag(cov)
            self.__R = cov / numpy.sqrt(numpy.dot(var[:, numpy.newaxis], var[numpy.newaxis, :]))
        return self.__R

    d = property(fget=getD, doc="dimension")
    p = property(fget=getP, doc="mean")
    R = property(fget=getR, doc="correlation")



def _pmf(gamma, param):
    '''
        Log probability mass function of the underlying log-linear model.
        @return random variable
    '''
    Beta = param['Beta']
    a = param['a']
    c = param['c']
    gamma = gamma[numpy.newaxis, :]
    L = numpy.empty(gamma.shape[0])
    for k in xrange(gamma.shape[0]):
        v = float(numpy.dot(numpy.dot(gamma, Beta), gamma.T))
        L[k] = (a + v) / c

    if gamma.shape[0] == 1: return L[0]
    else: return L

def tau(i, j):
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
    a = numpy.zeros(d * (d + 1) / 2)
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
    d = int((numpy.sqrt(1 + 8 * d) - 1) / 2)
    A = numpy.zeros((d, d))
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
    S = R * numpy.sqrt(numpy.dot(var[:, numpy.newaxis], var[numpy.newaxis, :])) + numpy.dot(p[:, numpy.newaxis], p[numpy.newaxis, :])
    S = 2 * S - numpy.diag(numpy.diag(S))

    # convert moments to vector
    s = m2v(S)

    # add normalization constant
    s = numpy.array(list(s) + [1.0])

    # solve moment equation
    x = scipy.linalg.solve(Z, s)

    return x[-1], v2m(x[:-1])





def random_problem(d, eps=0.05):
    '''
        Creates a random mean vector and correlation matrix that are consistent with the constraints on binary data.
        @param d dimension
        @param eps minmum distance to constraint limit
        @return p,R mean vector, correlation matrix
    '''
    p = eps + (1.0 - 2 * eps) * numpy.random.random(d)
    R = numpy.random.random((d, d))
    R = numpy.dot(R, R.T)

    for i in range(d):
        for j in range(i):
            low = max(0, p[i] + p[j] - 1)
            high = min(p[i], p[j]) - eps
            if not (low < R[i, j] < high):
                R[i, j] = min(high, max(low, R[i, j]))
                R[j, i] = R[i, j]
    V = R.diagonal()[numpy.newaxis, :]
    R = R / numpy.sqrt(numpy.dot(V.T, V))
    return p, R

def fit_logistic_model(d=6, n=5000):
    '''
        Constructs a linear binary model with given mean and correlations.
        Generates a (not necessarily random) weighted sample, where the weights
        are allowed to be negative. Fits a logistic model to the weighted sample.
        @param d dimension
        @param n sample size
        @todo The fact that the weights are partially negative cause the likelihood function
        to be not necessarily unimodal. Also, the concept of complete separation has to be
        adapted to the case of weighted samples. There is an analytical analysis to be done
        before trying to master the numerics.
    '''

    # Construct random problem.
    p, R = random_problem(d)

    # construct a linear binary model from p and R
    b = LinearBinary.from_moments(p, R)
    print utils.format.format(p, 'p')
    print utils.format.format(R, 'R')

    sample = utils.data.data()

    if n > 2 ** d:
        # Enumerate the state space.
        for dec in range(2 ** d):
            y = utils.format.dec2bin(dec, b.d)
            sample.append(y, b.pmf(y))
    else:
        # Sample states uniformly.
        logistic = binary.logistic_model.LogisticBinary.uniform(d)
        for i in range(n):
            y, logprob = logistic.rvslpmf()
            sample.append(y, b.pmf(y) / numpy.exp(logprob))

    logistic = binary.logistic_model.LogisticBinary.from_data(sample, eps=0.01, delta=0.01, verbose=True)
    print logistic.marginals()


def main():
    fit_logistic_model(d=6, n=5000)

if __name__ == "__main__":
    main()
