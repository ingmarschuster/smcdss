#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-10-29 20:13:19 +0200 (ven., 29 oct. 2010) $
    $Revision: 30 $
'''

__version__ = "$Revision: 30 $"


from numpy import random, dot, array
from binary import *
from auxpy.data import *

class PermanentBinary(Binary):
    '''

    '''
    def __init__(self, A):
        self.A = A
        self.d = A.shape[0]
        self.min = 1.0
        for i in range(self.d): self.min *= self.A[i, :].sum()

    def _pmf(self, gamma):
        prob = 1.0 - 2.0 * (gamma.sum() % 2)
        for i in range(self.d): prob *= dot(self.A[i, :], gamma)
        return max(prob, 0)

    def _lpmf(self, gamma):
        return log(self._pmf(gamma))

A = array(random.random((11, 11)), dtype=float)
pb = PermanentBinary(A)
print format(A)
sample = pb.marginals()
print sample
print sum(sample._W)
