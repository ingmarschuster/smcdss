#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import numpy

from utils.data import *
from binary import ProductBinary

class LogLinearBinary(ProductBinary):

    def __init__(self, A, p_0):
        '''
            Constructor.
            @param A coefficient matrix
        '''
        self.f_lpmf = _lpmf
        self.f_rvs = None
        self.f_rvslpmf = None
        self.param = dict(A=A, p_0=p_0)

    @classmethod
    def independent(cls, p):
        '''
            Constructs a log-linear-binary model with independent components.
            @param cls class 
            @param p mean
        '''
        d = p.shape[0]
        logOdds = numpy.log(p / (numpy.ones(d) - p))
        return cls(numpy.diag(logOdds), p[0])

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
            A[:i, i] = A[i, :i]

        # compute expected value of first component
        sample = data()
        for dec in range(2 ** d):
            bin = dec2bin(dec, d)
            prob = exp(float(numpy.dot(numpy.dot(bin[newaxis, :], A), bin[:, newaxis])))
            sample.append(bin, prob)
        p_0 = sample.getMean(weight=True)[0]

        return LogLinearBinary(A, p_0)

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

def calc_marginal(A, logc=0.0):
    '''
        Computes the parameters of a loglinear model where the last component has been marginalized.
        The marginalization is not exact but relies on an approximation idea by Cox and Wermuth
        [A note on the quadratic exponential binary distribution, Biometrika 1994, 81, 2, pp. 403-8].
        @param A coefficient matrix
        @param logc log normalization constant
        @return coefficient matrix the approximate marginal distribution
        @return log normalization constant of the approximate marginal distribution
        @todo The code needs to be extended such that not only the last but any component can
        be margined out approximately.
        '''
    d = A.shape[0]

    # normalization constant
    logc += numpy.log(1 + exp(A[d - 1, d - 1]))

    # coefficient matrix
    b = A[d - 1, :d - 1]
    A = (A[:d - 1, :d - 1] +
         (1 + tanh(0.5 * A[d - 1, d - 1])) * diag(b) +
          0.5 * sech(0.5 * A[d - 1, d - 1]) ** 2 * numpy.dot(b[:, newaxis], b[newaxis, :]))

    return A, logc
