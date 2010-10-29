#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from auxpy.data import *
from numpy import array, ones, zeros, log
from numpy.random import rand
from scipy.stats import rv_discrete

class ProductBinary(rv_discrete):
    '''
        A multivariable Bernoulli with independent components.
    '''
    def __init__(self, p=None, name='product-binary', longname='A multivariable Bernoulli with independent components.'):
        '''
            Constructor.
            @param p mean vector
            @param name name
            @param longname longname
        '''
        rv_discrete.__init__(self, name=name, longname=longname)
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

    def pmf(self, gamma):
        '''
            Probability mass function.
            @param gamma: binary vector
        '''
        if self.d == 0: return 1.0
        return self._pmf(gamma)

    def lpmf(self, gamma):
        '''
            Log-probability mass function.
            @param gamma binary vector
        '''
        if self.d == 0: return 0.0
        return self._lpmf(gamma)

    def rvs(self):
        '''
            Generates a random variable.
        '''
        if self.d == 0: return array([])
        return self._rvs()

    def rvslpmf(self):
        '''
            Generates a random variable and computes its likelihood.
        '''
        if self.d == 0: return array([]), 0.0
        return self._rvslpmf()


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

    def rvstest(self, n):
        '''
            Prints the empirical mean and correlation to stdout.
            @param n sample size
        '''
        sample = data()
        sample.sample(self, n)
        return format(sample.mean, 'sample (n = %i) mean' % n) + '\n' + \
               format(sample.cor, 'sample (n = %i) correlation' % n)

    def marginals(self):
        '''
            Get string representation of the marginals. 
            @remark Evaluation of the marginals requires exponential time. Do not do it.
            @return a string representation of the marginals 
        '''
        sample = data()
        for dec in range(2 ** self.d):
            bin = dec2bin(dec, self.d)
            sample.append(bin, self.pmf(bin))
        return str(sample)

    d = property(fget=getD, doc="dimension")
