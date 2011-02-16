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