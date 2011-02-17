#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from utils.data import *
from numpy import array, ones, zeros, log
from numpy.random import rand
from scipy.stats import rv_discrete

class Binary(rv_discrete):
    '''
        A multivariate Bernoulli.
    '''
    def __init__(self, name='binary', longname='A multivariate Bernoulli.'):
        '''
            Constructor.
            @param name name
            @param longname longname
        '''
        rv_discrete.__init__(self, name=name, longname=longname)

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

    def rvslpmf(self, n=None):
        '''
            Generates a random variable and computes its probability.
        '''
        if n is None: return self.rvslpmf()
        arr_Y = empty((n, self.d), dtype=bool)
        arr_log_f = empty(n, dtype=float)
        for index in xrange(n):
            arr_Y[index], arr_log_f[index] = self._rvslpmf()
        return arr_Y, arr_log_f

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
            sample.append(bin, self.lpmf(bin))
        return sample
