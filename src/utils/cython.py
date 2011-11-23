#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Cython import.
"""

"""
@namespace utils.cython
$Author$
$Rev$
$Date$
@details
"""

import utils
import numpy

def resample(w, u):
    return utils.cython_src.resample(w, u)


def uniform_lpmf(gamma, param):
    return utils.cython_src._uniform_all(param['q'], gamma=numpy.array(gamma, dtype=numpy.int8))[1]

def uniform_rvs(U, param):
    return utils.cython_src._uniform_all(param['q'], U=U)[0]

def uniform_rvslpmf(U, param):
    return utils.cython_src._uniform_all(param['q'], U=U)


def logistic_lpmf(gamma, param):
    return utils.cython_src._logistic_all(param['Beta'], gamma=numpy.array(gamma, dtype=numpy.int8))[1]

def logistic_rvs(U, param):
    return utils.cython_src._logistic_all(param['Beta'], U=U)[0]

def logistic_rvslpmf(U, param):
    return utils.cython_src._logistic_all(param['Beta'], U=U)
