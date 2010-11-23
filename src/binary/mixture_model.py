#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-10-29 20:13:19 +0200 (ven., 29 oct. 2010) $
    $Revision: 30 $
'''

__version__ = "$Revision: 30 $"

from auxpy.data import *
from binary import *
from numpy import random

class MixtureBinary(Binary):
    '''
        A mixture model consisting of a product model and a hybrid model.
    '''

    def __init__(self, dHybrid, lag):
        '''
            Constructor.
            @param dHybrid current hybrid model
            @param rProd ratio of the product model
            @param dProd current product model
            @param lagProd renewal lag for product model; governs how much the former model is still used
            @param lagHybrid renewal lag for hybrid model; governs how much the former model is still used
        '''

        Binary.__init__(self, name='mixture-binary', longname='A mixture model consisting of a product model and a hybrid model.')

        self.dHybridCurrent = dHybrid
        self.dHybridFormer = dHybrid
        self.lag = lag
        self.mean = 0.5 * ones(self.d)

    def renew_from_data(self, sample, fProd=1.0, fDep=1.0, eps=0.05, delta=0.1, verbose=False):
        '''
            Renews the mixture moving the current to the former and calibrating the current from the data.
            @param sample data
        '''

        self.dHybridFormer = self.dHybridCurrent
        self.mean = (1.0 - self.lag) * sample.getMean(weight=True, fraction=fProd) + self.lag * self.mean

        # construct hybrid model using smoothed mean
        self.dHybridCurrent = \
        self.dHybridCurrent.from_data(sample.fraction(fDep), eps, delta, mean=self.mean, verbose=verbose)

        # keep former hybrid model using smoothed mean
        self.dHybridFormer.dProd.p = self.mean[self.dHybridFormer.iProd]

    def _rvs(self):
        '''
            Generates a random variable.
        '''
        if random.random() < self.lag:
            # return from former model
            return self.dHybridFormer.rvs()
        else:
            # return from current model
            return self.dHybridCurrent.rvs()

    def _pmf(self, gamma):
        '''
            Probability mass function; works only if the Hybridendency model is normalized.
            @param gamma: binary vector
        '''
        pProd = self.dProdCurrent.pmf(gamma)
        pHybrid = (1.0 - self.lagHybrid) * self.dHybridCurrent.pmf(gamma)
        if self.lagHybrid > 0: pHybrid += self.lagHybrid * self.dHybridFormer.pmf(gamma)
        return (self.rProd) * pProd + (1.0 - self.rProd) * pHybrid

    def _lpmf(self, gamma):
        '''
            Log-probability mass function; works only if the hybrid model is normalized.
            @param gamma binary vector
        '''
        return log(self._pmf)

    def getD(self):
        '''
            Get dimension.
            @return dimension
        '''
        return self.dHybridCurrent.d

    d = property(fget=getD, doc="dimension")

