#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import numpy as np

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

def log_regr_rvs(Beta, u=None, gamma=None):
    d = Beta.shape[0]
    if u is not None:
        gamma = np.empty(d, dtype=bool)
        logu = np.log(u)

    logp = 0
    for i in xrange(0, d):
        # Compute log conditional probability that gamma(i) is one
        sum = Beta[i][i] + np.dot(Beta[i, 0:i], gamma[0:i])
        logcprob = -np.log(1 + np.exp(-sum))

        # Generate the ith entry
        if u is not None: gamma[i] = logu[i] < logcprob

        # Add to log conditional probability
        logp += logcprob
        if not gamma[i]: logp -= sum

    return gamma, logp
