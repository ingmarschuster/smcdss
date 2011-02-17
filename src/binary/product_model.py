#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

import numpy

import utils
import binary

class ProductBinary(binary.Binary):
    '''
        A binary model with independent components.
    '''
    def __init__(self, p=None, name='product-binary', longname='A binary model with independent components.'):
        ''' Constructor.
            @param p mean vector
            @param name name
            @param longname longname
        '''
        binary.Binary.__init__(self, name=name, longname=longname)
        if not p is None:
            if isinstance(p, (numpy.ndarray, list)):
                self.p = numpy.array(p, dtype=float)
            else:
                self.p = array([p])

        self.f_lpmf = _lpmf
        self.f_rvs = _rvs
        self.f_rvslpmf = _rvslpmf
        self.param = dict(p=p)

    def __str__(self):
        return utils.format.format_vector(self.p, 'p')

    @classmethod
    def random(cls, d):
        ''' Construct a random product model for testing.
            @param cls class
            @param d dimension
        '''
        return cls(numpy.random.random(d))

    @classmethod
    def uniform(cls, d):
        ''' Construct a random product model for testing.
            @param cls class
            @param d dimension
        '''
        return cls(p=0.5 * numpy.ones(d))

    @classmethod
    def from_data(cls, sample):
        ''' Construct a product model from data.
            @param cls class
            @param sample a sample of binary data
        '''
        return cls(sample.mean)

    def renew_from_data(self, sample, lag=0.0, verbose=False):
        ''' Updates the product model from data.
            @param cls class
            @param sample a sample of binary data
            @param lag lag
            @param verbose detailed information
        '''
        p = sample.getMean(weight=(sample.ess > 0.1))
        self.p = (1 - lag) * p + lag * self.p

    def getD(self):
        ''' Get dimension.
            @return dimension 
        '''
        return self.p.shape[0]

    d = property(fget=getD, doc="dimension")


def _lpmf(gamma, param):
    ''' Log-probability mass function.
        @param gamma binary vector
        @param param parameters
        @return log-probabilities
    '''
    p = param['p']
    L = numpy.empty(gamma.shape[0])
    for k in xrange(gamma.shape[0]):
        prob = 1.0
        for i, m in enumerate(p):
            if gamma[k, i]: prob *= m
            else: prob *= (1 - m)
        L[k] = prob
    return numpy.log(L)

def _rvs(U, param):
    ''' Generates a random variable.
        @param U uniform variables
        @param param parameters
        @return binary variables
    '''
    p = param['p']
    Y = numpy.empty((U.shape[0], U.shape[1]), dtype=bool)
    for k in xrange(U.shape[0]):
        Y[k] = p > U[k]
    return Y

def _rvslpmf(U, param):
    ''' Generates a random variable and computes its probability.
        @param U uniform variables
        @param param parameters
        @return binary variables, log-probabilities
    '''
    Y = _rvs(U, param)
    return Y, _lpmf(Y, param)
