#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2011-02-17 18:37:17 +0100 (jeu., 17 févr. 2011) $
    $Revision: 74 $
'''

import utils
import numpy

def resample(w, u):
    return utils.cython_src.resample(w, u)

def logistic_lpmf(gamma, param):
    return utils.cython_src._logistic_all(param['Beta'], gamma=numpy.array(gamma, dtype=numpy.int8))[1]

def logistic_rvs(U, param):
    return utils.cython_src._logistic_all(param['Beta'], U=U)[0]

def logistic_rvslpmf(U, param):
    return utils.cython_src._logistic_all(param['Beta'], U=U)
