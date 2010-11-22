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

    def __init__(self, dHybrid, rProd=0.0, dProd=None, lagProd=0.0, lagHybrid=0.0):
        '''
            Constructor.
            @param dHybrid current hybrid model
            @param rProd ratio of the product model
            @param dProd current product model
            @param lagProd renewal lag for product model; governs how much the former model is still used
            @param lagHybrid renewal lag for hybrid model; governs how much the former model is still used
        '''

        Binary.__init__(self, name='mixture-binary', longname='A mixture model consisting of a product model and a hybrid model.')

        self.rProd = rProd
        self.lagProd = lagProd
        self.lagHybrid = lagHybrid
        self.dHybridCurrent = dHybrid
        self.dHybridFormer = dHybrid
        if dProd is None: dProd = ProductBinary.uniform(self.d)
        self.dProdCurrent = dProd

    def renew(self, dProdNew, dHybridNew):
        '''
            Renews the mixture changing the new to the current, and the current to the former.
            @param dProdNew new product model
            @param dHybridNew new hybrid model
        '''
        self.dProdCurrent = ProductBinary(p=(1.0 - self.lagProd) * dProdNew.p + self.lagProd * self.dProdCurrent.p)
        self.dHybridFormer = self.dHybridCurrent
        self.dHybridCurrent = dHybridNew

    def renew_from_data(self, sample, fProd=1.0, fHybrid=1.0, eps=0.05, delta=0.1):
        '''
            Renews the mixture moving the current to the former and calibrating the current from the data.
            @param sample data
        '''
        dProdNew = self.dProdCurrent.from_data(sample.fraction(fProd))
        dHybridNew = self.dHybridCurrent.from_data(sample.fraction(fHybrid), eps, delta)
        self.renew(dProdNew, dHybridNew)

    def _rvs(self):
        '''
            Generates a random variable.
        '''
        if random.random() < self.rProd:
            # return inHybridendent components
            return self.dProdCurrent.rvs()

        if random.random() < self.lagProd:
            # return from old model
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

