#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import numpy as np
import utils

def resample(w, u):
    '''
        Computes the particle indices by systematic resampling.
        @param w array of weights
    '''
    n = w.shape[0]
    cnw = n * np.cumsum(w)
    j = 0
    i = np.empty(n, dtype="int")
    for k in xrange(n):
        while cnw[j] < u:
            j = j + 1
        i[k] = j
        u = u + 1.
    return i

def _logistic_all(param, U=None, gamma=None):
    ''' Generates a random variable.
        @param U uniform variables
        @param param parameters
        @return binary variables
    '''
    Beta = param['Beta']
    if U is not None:
        size = U.shape[0]
        d = U.shape[1]
        gamma = np.empty((size, U.shape[1]), dtype=bool)
        logU = np.log(U)

    if gamma is not None:
        size = gamma.shape[0]
        d = gamma.shape[1]

    L = np.zeros(size, dtype=float)

    for k in xrange(size):

        for i in xrange(d):
            # Compute log conditional probability that gamma(i) is one
            sum = Beta[i, i] + np.dot(Beta[i, 0:i], gamma[k, 0:i])
            logcprob = -np.log(1 + np.exp(-sum))

            # Generate the ith entry
            if U is not None: gamma[k, i] = logU[k, i] < logcprob

            # Add to log conditional probability
            L[k] += logcprob
            if not gamma[k, i]: L[k] -= sum

    return gamma, L

def logistic_lpmf(gamma, param):
    return utils.python._logistic_all(param, gamma=gamma)[1]

def logistic_rvs(U, param):
    return utils.python._logistic_all(param, U=U)[0]

def logistic_rvslpmf(U, param):
    return utils.python._logistic_all(param, U=U)
