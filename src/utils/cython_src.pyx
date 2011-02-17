#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double exp(double)
    double log(double)

def resample(np.ndarray[dtype=np.float64_t, ndim=1] w, double u):
   '''
       Computes the particle indices by systematic resampling.
       @param w array of weights
   '''
   w = w * w.shape[0]
   cdef int j = 0
   cdef int k
   cdef double cumsum = w[0]
   cdef np.ndarray[dtype=np.int_t, ndim=1] i = np.zeros(w.shape[0], dtype=np.int)
   for k in xrange(w.shape[0]):
       while cumsum < u:
           j = j + 1
           cumsum += w[j]
       i[k] = j
       u = u + 1.0
   return i


def _logistic_all(np.ndarray[dtype=np.float64_t, ndim=2] Beta,
                  np.ndarray[dtype=np.float64_t, ndim=2] U=None,
                  np.ndarray[dtype=np.int8_t, ndim=2] gamma=None):
    ''' Generates a random variable.
        @param U uniform variables
        @param param parameters
        @return binary variables
    '''
    cdef int d = Beta.shape[0]
    cdef int i, size
    cdef double logp = 0.0
    cdef double sum
 
    if U is not None:
        size = U.shape[0]
        gamma = np.empty((size, U.shape[1]), dtype=np.int8)
    
    if gamma is not None:
        size = gamma.shape[0]

    L = np.zeros(size, dtype=np.float64)
    
    for k in xrange(size):

        for i in xrange(0, d):
            # Compute log conditional probability that gamma(i) is one
            sum = Beta[i, i]
            for j in xrange(i):
                sum += Beta[i, j] * gamma[k,j]
            logcprob = -log(1 + exp(-sum))
    
            # Generate the ith entry
            if U is not None: gamma[k,i] = log(U[k,i]) < logcprob
    
            # Add to log conditional probability
            L[k] += logcprob
            if not gamma[k,i]: L[k] -= sum

    return np.array(gamma, dtype=bool), L