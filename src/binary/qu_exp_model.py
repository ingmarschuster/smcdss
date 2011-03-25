#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import numpy
from numpy import *
import scipy.stats

import binary
import utils

class QuExpBinary(binary.ProductBinary):


    def __init__(self, A, name='quadratic exponential binary', longname='A quadratic exponential binary model.'):
        ''' Constructor.
            @param A matrix of coefficients
        '''

        binary.ProductBinary.__init__(self, name=name, longname=longname)
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
        p = random.random(d)
        logratio = numpy.log(p / (1 - p))
        A = diag(logratio)
        for i in range(d):
            if scale > 0.0: A[i, :i] = random.normal(scale=scale, size=i)

        return QuExpBinary(A)

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.param['A'].shape[0]

    d = property(fget=getD, doc="dimension")

def _lpmf(gamma, param):
    '''
        Log probability mass function of the underlying log-linear model.
        @return random variable
    '''
    A = param['A']
    L = numpy.empty(gamma.shape[0])
    for k in xrange(gamma.shape[0]):
        L[k] = float(numpy.dot(numpy.dot(gamma, param['A']), gamma.T))
    return L

def sech(x):
    '''
        Hyperbolic secant.
        @param x value
        @return sech(x)
    '''
    return 1 / cosh(x)

#def calc_marginal(A, logc=0.0):
#    '''
#        Computes the parameters of a loglinear model where the last component has been marginalized.
#        The marginalization is not exact but relies on an approximation idea by Cox and Wermuth
#        [A note on the quadratic exponential binary distribution, Biometrika 1994, 81, 2, pp. 403-8].
#        @param A coefficient matrix
#        @param logc log normalization constant
#        @return coefficient matrix the approximate marginal distribution
#        @return log normalization constant of the approximate marginal distribution
#        @todo The code needs to be extended such that not only the last but any component can
#        be margined out approximately.
#        '''
#    d = A.shape[0]
#
#    # normalization constant
#    logc += numpy.log(1 + exp(A[d - 1, d - 1]))
#
#    # coefficient matrix
#    b = A[d - 1, :d - 1]
#    A = (A[:d - 1, :d - 1] +
#         (1 + tanh(0.5 * A[d - 1, d - 1])) * diag(b) +
#          0.5 * sech(0.5 * A[d - 1, d - 1]) ** 2 * numpy.dot(b[:, newaxis], b[newaxis, :]))
#
#    return A, logc

#xi = [0, 0.5 * tanh(0.5 * A[k, k]), 0.125 * sech(0.5 * A[k, k]) ** 2]

def calc_marginal(A):
    '''
    '''

    I = range(A.shape[0])
    d = len(I)

    num = 10
    w = empty(num)
    p = 0.5
    for j in xrange(num):
        w[j] = scipy.stats.binom.pmf(j, num - 1, p)
    w = sqrt(w)
    #w=ones(num)

    X = ones((num, 3))
    Beta = numpy.zeros((d, d))
    perm = list()

    while d > 0:
        l, u, B = empty(d), empty(d), empty((d, d - 1))

        # determine range of each component
        for k in range(d):
            B[k] = concatenate((A[k, :k], A[k + 1:, k]))
            l[k] = max(B[k][B[k] < 0].sum() + A[k, k], -15)
            u[k] = min(B[k][B[k] > 0].sum() + A[k, k], 15)

        # compute least square regression and error
        xi, eps = list(), list()
        for k in range(d):
            if u[k] - l[k] > 0:
                x = linspace(l[k], u[k], num=num)
                y = w * log(cosh(x))
                X[:, 1] = x
                X[:, 2] = power(x, 2)
                W = w[:, newaxis] * X
                xi.append(scipy.linalg.solve(dot(W.T, W), dot(W.T, y)))
                eps.append(numpy.linalg.norm(dot(W, xi[-1]) - y))
            else:
                xi.append(zeros(3))
                eps.append(0)

        # pick component with least error
        k = eps.index(min(eps))
        xi = xi[k]
        # xi = [0, 0.5 * tanh(0.5 * A[k, k]), 0.125 * sech(0.5 * A[k, k]) ** 2]

        r = range(k) + range(k + 1, d)
        b = concatenate((B[k], array([A[k, k]])))
        A = A[r, :][:, r] + xi[1] * diag(B[k]) + (0.5 + xi[2]) * tril(outer(B[k], B[k]))

        perm.append(I[k])
        Beta[d - 1, :d] = b

        I.pop(k)
        d = len(I)

    model = binary.LogisticBinary(Beta)
    perm.reverse()
    model.v2m_perm = perm
    return model

def calc_logistic_model(A):

        d = A.shape[0]
        Beta = numpy.zeros((d, d))
        perm = list()

        for i in xrange(d - 1, -1, -1):
            A, b, k = calc_marginal(A)
            Beta[i, :i + 1] = b
            perm.append(k)

        b = binary.LogisticBinary(Beta)
        b.m2v_perm = perm
        return b

def main():
    d = 5
    #A = tril(array([1, 0, 0, -1, -.2, 0, .2, -.3, 1]).reshape((3, 3)))
    #a = QuExpBinary.random(d, scale=.5)
    #A = a.param['A']

    A = utils.ubqp.load_ubqp() / 100.0
    b = calc_marginal(A)

    d = utils.data.data()
    d.sample(f=b, size=5000)
    r = range(15)
    F = d.cor[r, :][:, r]
    #F = A
    print utils.format.format(F)
    print utils.format.format(d.mean)


if __name__ == "__main__":
    main()
