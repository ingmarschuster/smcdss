#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from auxpy.data import *
from numpy import array, ones, zeros, log, concatenate
from numpy.random import rand
from binary import Binary

class ProductBinary(Binary):
    '''
        A multivariate Bernoulli with independent components.
    '''
    def __init__(self, p=None, name='product-binary', longname='A multivariate Bernoulli with independent components.'):
        '''
            Constructor.
            @param p mean vector
            @param name name
            @param longname longname
        '''
        Binary.__init__(self, name=name, longname=longname)
        if not p is None:
            if isinstance(p, (ndarray, list)):
                self.p = array(p, dtype=float)
            else:
                self.p = array([p])

    def __str__(self):
        return format_vector(self.p, 'p')

    @classmethod
    def random(cls, d):
        '''
            Construct a random product-binary model for testing.
            @param cls class
            @param d dimension
        '''
        return cls(rand(d))

    @classmethod
    def uniform(cls, d):
        '''
            Construct a random product-binary model for testing.
            @param cls class
            @param d dimension
        '''
        return cls(0.5 * ones(d))

    @classmethod
    def from_data(cls, sample):
        '''
            Construct a product-binary model from data.
            @param cls class
            @param sample a sample of binary data
        '''
        return cls(sample.mean)

    def _pmf(self, gamma):
        '''
            Probability mass function.
            @param gamma: binary vector
        '''
        prob = 1.
        for i, mprob in enumerate(self.p):
            if gamma[i]: prob *= mprob
            else: prob *= (1 - mprob)
        return prob

    def _lpmf(self, gamma):
        '''
            Log-probability mass function.
            @param gamma binary vector
        '''
        return log(self.pmf(gamma))

    def _rvs(self):
        '''
            Generates a random variable.
        '''
        if self.d == 0: return []
        return self.p > rand(self.d)

    def _rvslpmf(self):
        '''
            Generates a random variable and computes its likelihood.
        '''
        rv = self.rvs()
        return rv, self.lpmf(rv)

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.p.shape[0]

    d = property(fget=getD, doc="dimension")
