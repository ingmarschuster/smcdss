#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-10-29 13:41:14 +0200 (ven., 29 oct. 2010) $
    $Revision: 29 $
'''

from numpy import *
from auxpy.data import *
from binary import ProductBinary

class LogLinearBinary(ProductBinary):

    def __init__(self, Beta):
        self.Beta = Beta

    @classmethod
    def independent(cls, p):
        '''
            Constructs a log-linear-binary model with independent components.
            @param cls class 
            @param p mean
        '''
        d = p.shape[0]
        logOdds = log(p / (ones(d) - p))
        return cls(diag(logOdds))

    @classmethod
    def random(cls, d):
        '''
            Constructs a random log-linear-binary model for testing.
            @param cls class 
            @param d dimension
        '''
        Beta = random.normal(scale=1.0, size=(d, d))
        for i in range(d):
            Beta[i, :i] = Beta[:i, i]
        return LogLinearBinary(Beta)

    def _pmf(self, gamma):
        '''
            Probability mass function.
            @param gamma: binary vector
        '''
        return exp(self._lpmf(gamma))

    def _lpmf(self, gamma):
        '''
            Log-probability mass function.
            @param gamma binary vector    
        '''
        return float(dot(dot(gamma[newaxis, :], self.Beta), gamma[:, newaxis]))

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.Beta.shape[0]

    d = property(fget=getD, doc="dimension")

# Computes the log-linear approximately marginalized over the dim - len(b) components.
def marginal_loglinear(b):
    B = range(len(b))
    C = range(len(b), DIM)
    
    mu = MU + len(C) * log(2)
    for r in C:
        mu += log(cosh(ALPHA[r, r]))
        for j in B:
              mu += .5 * ALPHA[j, r] ** 2 * cosh(ALPHA[r, r]) ** -2

    alpha = copy(ALPHA)[:len(B), :len(B)]
    for j in B:
        for r in C:
            alpha[j, j] += ALPHA[j, r] * tanh(ALPHA[r, r]) 
            for s in C:
                  if r > s:
                      alpha[j, j] += ALPHA[j, r] * ALPHA[r, s] * tanh(ALPHA[s, s]) * cosh(ALPHA[r, r]) ** -2
    for j in B:
        for k in B:
            if j > k:
                for r in C:
                    alpha[j, k] += ALPHA[j, r] * ALPHA[k, r] * cosh(ALPHA[r, r]) ** -2
    
    sum = dot(diag(alpha), b)
    for i in B:
        for j in range(i):
            sum += b[i] * b[j] * alpha[i, j]
    return exp(MU + sum)