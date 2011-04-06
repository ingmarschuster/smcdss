#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from binary import *

class QuExpBinary(ProductBinary):


    def __init__(self, A, name='quadratic exponential binary', longname='A quadratic exponential binary model.'):
        ''' Constructor.
            @param A matrix of coefficients
        '''

        ProductBinary.__init__(self, name=name, longname=longname)
        self.f_lpmf = _lpmf
        self.f_rvs = None
        self.f_rvslpmf = None
        self.param = dict(A=A)

    @classmethod
    def independent(cls, p):
        '''
            Constructs a log-linear-binary model with independent components.
            @param cls class 
            @param p mean
        '''
        d = p.shape[0]
        logOdds = numpy.log(p / (numpy.ones(d) - p))
        return cls(numpy.diag(logOdds))

    @classmethod
    def random(cls, d, scale=0.5):
        '''
            Constructs a random log-linear-binary model for testing.
            @param cls class 
            @param d dimension
            @param scale standard deviation of the off-diagonal elements
        '''
        p = numpy.random.random(d)
        logratio = numpy.log(p / (1 - p))
        A = numpy.diag(logratio)
        for i in range(d):
            if scale > 0.0: A[i, :i] = numpy.random.normal(scale=scale, size=i)

        return QuExpBinary(A)

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.A.shape[0]

    def getA(self):
        return self.param['A']

    A = property(fget=getA, doc="A")

def _lpmf(gamma, param):
    '''
        Log probability mass function of the underlying quadratic exponential model.
        @return random variable
    '''
    L = numpy.empty(gamma.shape[0])
    for k in xrange(gamma.shape[0]):
        L[k] = float(numpy.dot(numpy.dot(gamma[k], param['A']), gamma[k].T))
    return L

def calc_marginal(A):
    '''
        Computes a quadratic exponential model where the component that causes the least
        quadratic approximation error has been marginalized. The method is inspired by Cox and Wermuth,
        A note on the quadratic exponential binary distribution, Biometrika 1994, 81, 2, pp. 403-8.
        @param A coefficient matrix
        @return model the approximate marginal distribution
    '''

    I = range(A.shape[0])
    d = len(I)

    num = 10
    w = numpy.empty(num)
    p = 0.5
    for j in xrange(num):
        w[j] = stats.binom.pmf(j, num - 1, p)
    w = numpy.sqrt(w)

    X = numpy.ones((num, 3))
    Beta = numpy.zeros((d, d))
    perm = list()

    while d > 0:
        l, u, B = numpy.empty(d), numpy.empty(d), numpy.empty((d, d - 1))

        # determine range of each component
        for k in range(d):
            B[k] = numpy.concatenate((A[k, :k], A[k + 1:, k]))
            l[k] = max(B[k][B[k] < 0].sum() + A[k, k], -15)
            u[k] = min(B[k][B[k] > 0].sum() + A[k, k], 15)

        # compute least square regression and error
        xi, eps = list(), list()
        for k in range(d):
            if u[k] - l[k] > 0:
                x = numpy.linspace(l[k], u[k], num=num)
                y = w * numpy.log(numpy.cosh(x))
                X[:, 1] = x
                X[:, 2] = numpy.power(x, 2)
                W = w[:, numpy.newaxis] * X
                xi.append(scipy.linalg.solve(numpy.dot(W.T, W), numpy.dot(W.T, y)))
                eps.append(numpy.linalg.norm(numpy.dot(W, xi[-1]) - y))
            else:
                xi.append(numpy.zeros(3))
                eps.append(0)

        # pick component with least error
        k = eps.index(min(eps))
        xi = xi[k]

        # the original Taylor series approach corresponds to 
        # xi = [0, 0.5 * numpy.tanh(0.5 * A[k, k]), 0.125 / (numpy.cosh(0.5 * A[k, k])** 2)]

        r = range(k) + range(k + 1, d)
        b = numpy.concatenate((B[k], numpy.array([A[k, k]])))
        A = A[r, :][:, r] + xi[1] * numpy.diag(B[k]) + (0.5 + xi[2]) * numpy.tril(numpy.outer(B[k], B[k]))

        perm.append(I[k])
        Beta[d - 1, :d] = b

        I.pop(k)
        d = len(I)

    model = logistic_cond_model.LogisticBinary(Beta)
    perm.reverse()
    model.v2m_perm = perm
    return model

def calc_logistic_model(A):
    '''
        Constructs a logistic conditional model from a quadratic exponential model by
        repeatedly computing the approximate marginal distributions
        @param A coefficient matrix
        @return model a logistic conditional model
    '''
    d = A.shape[0]
    Beta = numpy.zeros((d, d))
    perm = list()

    for i in xrange(d - 1, -1, -1):
        A, b, k = calc_marginal(A)
        Beta[i, :i + 1] = b
        perm.append(k)

    model = logistic_cond_model.LogisticBinary(Beta)
    model.m2v_perm = perm
    return model

def main():
    pass

if __name__ == "__main__":
    main()
