'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from auxpy.data import *
from binary import *

class HybridBinary(ProductBinary):
    '''
        A hybrid model having constant, independent and dependent components.
    '''
    
    def __init__(self, cBase, iProd, iDep, dProd, dDep):
        '''
            Constructor.
            @param cBase base vector with set constant components
            @param iProd index of independent components
            @param iDep index of dependent components
            @param dProd product model
            @param dDep binary model supporting dependencies
        '''

        ## base vector with set constant components
        self.__cBase = cBase
        ## index of independent components
        self.__iProd = list(iProd)
        ## index of dependent components
        self.__iDep = list(iDep)
        ## index of constant components
        self.__iConst = [i for i in range(self.d) if not i in iProd + iDep]
        ## product model
        self.dProd = dProd
        ## binary model supporting dependencies
        self.dDep = dDep

    def __str__(self):
        return 'constant: ' + str(self.iConst) + '\n' + format(self.__cBase[self.iConst], 'base vector') + '\n' + \
               'product model: ' + str(self.iProd) + '\n' + str(self.dProd) + '\n' + \
               'dependency model: ' + str(self.iDep) + '\n' + str(self.dDep) + '\n'

    @classmethod
    def from_data(cls, sample, eps=0.05, delta=0.1, model=LogisticRegrBinary):
        '''
            Construct a hybrid-binary model from data.
            @param cls class
            @param eps a component is constant if the distance of then mean from the boundaries of [0,1] is less than minEps
            @param delta a component is independent if the total correlation is less than minDelta
            @param model binary model for dependencies
            @param sample a sample of binary data
        '''

        # compute mean
        mean = sample.getMean(weight=True)
        cBase = mean > 0.5

        # random components
        boolRand = (mean > eps) * (mean < 1 - eps)

        # compute 1/2-norm of correlation coefficients
        acor = calc_norm(sample.cor - eye(mean.shape[0]), 0.5) / float(boolRand.sum())

        # classify random components
        boolDep = (acor > delta) * boolRand
        boolProd = (boolDep ^ True) * boolRand

        # init sub-models
        iProd = list(where(boolProd == True)[0])
        iDep = list(where(boolDep == True)[0])
        dProd = ProductBinary(mean[iProd])
        dDep = model.fromData(sample.getSubData(iDep))

        return cls(cBase, iProd, iDep, dProd, dDep)

    def _pmf(self, gamma):
        '''
            Probability mass function.
            @param gamma: binary vector
        '''
        if not ((self.__cBase == gamma)[self.iConst]).all(): return 0
        return self.dProd.pmf(gamma[self.iProd]) * self.dDep.pmf(gamma[self.iDep])

    def _lpmf(self, gamma):
        '''
            Log-probability mass function.
            @param gamma binary vector
        '''
        if not ((self.__cBase == gamma)[self.iConst]).all(): return - inf
        return self.dProd.lpmf(gamma[self.iProd]) + self.dDep.lpmf(gamma[self.iDep])

    def _rvs(self):
        '''
            Generates a random variate.
        '''
        rv = self.__cBase.copy()
        rv[self.iProd] = self.dProd.rvs()
        rv[self.iDep] = self.dDep.rvs()

        return rv

    def _rvslpmf(self):
        '''
            Generates a random variate and computes its likelihood.
        '''
        rv = self.__cBase.copy()
        rvProd, lmpfProd = self.dProd.rvsplus()
        rv[self.iProd] = rvProd
        rvDep, lmpfDep = self.dDep.rvsplus()
        rv[self.iDep] = rvDep

        return rv, lmpfProd + lmpfDep

    def getIProd(self):
        '''
            Get index of independent components.
            @return index
        '''
        return self.__iProd


    def getIDep(self):
        '''
            Get index of dependent components.
            @return index
        '''
        return self.__iDep


    def getIConst(self):
        '''
            Get index of constant components.
            @return index
        '''
        return self.__iConst

    def getIRandom(self):
        '''
            Get index of random components.
            @return index
        '''
        return self.__iProd + self.__iDep

    def getIZeros(self):
        '''
            Get index of constant zero (False) components.
            @return index
        '''
        return [i for i in self.__iConst if i in where(self.__cBase ^ True)[0]]

    def getIOnes(self):
        '''
            Get index of constant one (True) components.
            @return index
        '''
        return [i for i in self.__iConst if i in where(self.__cBase ^ False)[0]]

    def getNProd(self):
        '''
            Get number of independent components.
            @return number
        '''
        return len(self.__iProd)

    def getNDep(self):
        '''
            Get number of dependent components.
            @return number
        '''
        return len(self.__iDep)


    def getNConst(self):
        '''
            Get number of constant components.
            @return number
        '''
        return len(self.__iConst)

    def getNRandom(self):
        '''
            Get number of random components.
            @return number
        '''
        return self.d - len(self.__iConst)

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
        return self.__cBase.shape[0]

    iProd = property(fget=getIProd, doc="index of independent components")
    iDep = property(fget=getIDep, doc="index of dependent components")
    iConst = property(fget=getIConst, doc="index of constant components")
    iRand = property(fget=getIRandom, doc="index of random components")
    iZeros = property(fget=getIZeros, doc="index of constant zero (False) components")
    iOnes = property(fget=getIOnes, doc="index of constant one (True) components")

    nProd = property(fget=getNProd, doc="number of independent components")
    nDep = property(fget=getNDep, doc="number of dependent components")
    nConst = property(fget=getNConst, doc="number of constant components")
    nRand = property(fget=getNRandom, doc="number of random components")
    nZeros = property(fget=getNZeros, doc="number of constant zero (False) components")
    nOnes = property(fget=getNOnes, doc="number of constant one (True) components")

    d = property(fget=getD, doc="dimension")
