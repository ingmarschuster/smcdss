#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

from numpy import *
from utils.data import *
from binary import ProductBinary

class LogLinearBinary(ProductBinary):

    def __init__(self, A, p_0):
        '''
            Constructor.
            @param A coefficient matrix
        '''
        self.A = A
        self.p_0 = p_0

    @classmethod
    def independent(cls, p):
        '''
            Constructs a log-linear-binary model with independent components.
            @param cls class 
            @param p mean
        '''
        d = p.shape[0]
        logOdds = log(p / (ones(d) - p))
        return cls(diag(logOdds), p[0])

    @classmethod
    def random(cls, d, scale=0.5):
        '''
            Constructs a random log-linear-binary model for testing.
            @param cls class 
            @param d dimension
            @param scale standard deviation of the off-diagonal elements
        '''
        p = random.random(d)
        logratio = log(p / (1 - p))
        A = diag(logratio)
        for i in range(d):
            if scale > 0.0: A[i, :i] = random.normal(scale=scale, size=i)
            A[:i, i] = A[i, :i]

        # compute expected value of first component
        sample = data()
        for dec in range(2 ** d):
            bin = dec2bin(dec, d)
            prob = exp(float(dot(dot(bin[newaxis, :], A), bin[:, newaxis])))
            sample.append(bin, prob)
        p_0 = sample.getMean(weight=True)[0]

        return LogLinearBinary(A, p_0)

    @classmethod
    def from_data(cls, sample):
        '''
            Construct a log-linear-binary model from data.
            @param cls class
            @param sample a sample of binary data
        '''
        return cls(calc_A(sample), sample.getMean(weight=True)[0])

    def _pmf(self, gamma):
        '''
            Probability mass function of the underlying log-linear model.
            @return random variable
        '''
        return exp(self._lpmf(gamma))

    def _lpmf(self, gamma):
        '''
            Log probability mass function of the underlying log-linear model.
            @return random variable
        '''
        gamma = gamma[newaxis, :]
        return float(dot(dot(gamma, self.A), gamma.T))

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.A.shape[0]

    d = property(fget=getD, doc="dimension")


def calc_A(sample):
    cor = sample.getCor(weight=True)
    
    return A

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
        '''
    d = A.shape[0]

    # normalization constant
    logc += log(1 + exp(A[d - 1, d - 1]))

    # coefficient matrix
    b = A[d - 1, :d - 1]
    A = (A[:d - 1, :d - 1] +
         (1 + tanh(0.5 * A[d - 1, d - 1])) * diag(b) +
          0.5 * sech(0.5 * A[d - 1, d - 1]) ** 2 * dot(b[:, newaxis], b[newaxis, :]))

    return A, logc
