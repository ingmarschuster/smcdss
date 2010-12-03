#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from auxpy.data import *
from binary import *

class HybridBinary(Binary):
    '''
        A hybrid model having constant and random components.
    '''

    def __init__(self, Const, Model, iModel, p, name='hybrid-binary', longname='A hybrid model having constant and random components.'):
        '''
            Constructor.
            @param Const base vector with set constant components
            @param iProd index of independent components
            @param iModel index of dependent components
            @param dProd product model
            @param Model binary model supporting dependencies
        '''

        ## base vector with set constant components
        self._Const = Const
        ## binary model
        self._Model = Model
        ## index of random components
        self._iModel = list(iModel)
        ## index of constant components
        self._iConst = [i for i in range(self.d) if not i in iModel]
        ## mean
        self.p = p

        Binary.__init__(self, 'hybrid-' + self._Model.name, longname)

    def __str__(self):
        return 'constant: ' + str(self.iConst) + '\n' + format(self._Const[self.iConst], 'base vector') + '\n' + \
               'random: ' + str(self.iModel) + '\n' + str(self._Model) + '\n'

    @classmethod
    def uniform(cls, d, model=LogisticRegrBinary):
        return cls(Const=zeros(d, dtype=bool),
                   Model=model.uniform(d), iModel=range(d),
                   p=0.5 * ones(d, dtype=float))

    @classmethod
    def from_data(cls, sample, min_d=0.0, model=LogisticRegrBinary, verbose=False):
        '''
            Construct a hybrid-binary model from data.
            @param cls class
            @param eps a component is constant if the distance of then mean from the boundaries of [0,1] is less than minEps
            @param delta a component is independent if the total correlation is less than minDelta
            @param model binary model for dependencies
            @param sample a sample of binary data
        '''

        # compute mean
        p = sample.getMean(weight=True)

        # set base vector and random components
        Const = p > 0.5
        iModel = list(where((mean >= min_d) * (mean <= 1 - min_d))[0])

        # initialize binary model
        Model = model.from_data(sample.get_sub_data(iModel), verbose=verbose)

        return cls(Const, Model, iModel, p)

    def renew_from_data(self, sample, verbose=False, **param):

        # keep previous parameters
        prvP = self.p
        prvIndex = self._iModel

        # compute mean
        if 'lag' in param.keys():
            lag = param['lag']
        else:
            lag = 0
        self.p = (1 - lag) * sample.mean + lag * self.p

        # base vector and index sets
        self._Const = self.p > 0.5

        # Determine constant components.
        self._iModel = []; self._iConst = []
        for i, prob in enumerate(self.p):
            if prob < param['min_d'] or prob > 1.0 - param['min_d']:
                self._iConst.append(i)
            else:
                self._iModel.append(i)
        adjIndex = self._iModel

        self._Model.renew_from_data(sample, prvIndex, adjIndex, lag=lag,
                                    eps=param['eps'], delta=param['delta'], prvP=prvP, verbose=verbose)

    def _pmf(self, gamma):
        '''
            Probability mass function.
            @param gamma: binary vector
        '''
        if not ((self._Const == gamma)[self.iConst]).all(): return 0
        return self._Model.pmf(gamma[self.iModel])

    def _lpmf(self, gamma):
        '''
            Log-probability mass function.
            @param gamma binary vector
        '''
        if not ((self._Const == gamma)[self.iConst]).all(): return - inf
        return self._Model.lpmf(gamma[self.iModel])

    def _rvs(self):
        '''
            Generates a random variable.
        '''
        rv = self._Const.copy()
        if self.nModel > 0: rv[self.iModel] = self._Model._rvs()
        return rv

    def _rvslpmf(self):
        '''
            Generates a random variable and computes its probability.
        '''
        rv = self._Const.copy()
        x, lmpf = self._Model._rvslpmf()
        rv[self.iModel] = x
        return rv, lmpf

    def getIModel(self):
        '''
            Get index of dependent components.
            @return index
        '''
        return self._iModel


    def getIConst(self):
        '''
            Get index of constant components.
            @return index
        '''
        return self._iConst

    def getIZeros(self):
        '''
            Get index of constant zero (False) components.
            @return index
        '''
        return [i for i in self._iConst if i in where(self._Const ^ True)[0]]

    def getIOnes(self):
        '''
            Get index of constant one (True) components.
            @return index
        '''
        return [i for i in self._iConst if i in where(self._Const ^ False)[0]]

    def getNModel(self):
        '''
            Get number of dependent components.
            @return number
        '''
        return len(self._iModel)

    def getNConst(self):
        '''
            Get number of constant components.
            @return number
        '''
        return len(self._iConst)

    def getNZeros(self):
        '''
            Get number of constant zero (False) components.
            @return number
        '''
        return len(self.iZeros)

    def getNOnes(self):
        '''
            Get number of constant one (True) components.
            @return number
        '''
        return len(self.iOnes)

    def getD(self):
        '''
            Get dimension.
            @return dimension
        '''
        return self._Const.shape[0]

    iModel = property(fget=getIModel, doc="index of dependent components")
    iConst = property(fget=getIConst, doc="index of constant components")
    iZeros = property(fget=getIZeros, doc="index of constant zero (False) components")
    iOnes = property(fget=getIOnes, doc="index of constant one (True) components")

    nModel = property(fget=getNModel, doc="number of dependent components")
    nConst = property(fget=getNConst, doc="number of constant components")
    nZeros = property(fget=getNZeros, doc="number of constant zero (False) components")
    nOnes = property(fget=getNOnes, doc="number of constant one (True) components")

    d = property(fget=getD, doc="dimension")
