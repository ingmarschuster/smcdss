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

# workaround for cython accepts has no numpy datatype bool_t
def log_regr_rvs(Beta, u=None, gamma=None):
    if not gamma is None: gamma = np.array(gamma, dtype=np.int8)
    return _log_regr_rvs(Beta, u=u, gamma=gamma)

def _log_regr_rvs(np.ndarray[dtype=np.float64_t, ndim=2] Beta,
                  np.ndarray[dtype=np.float64_t, ndim=1] u=None,
                  np.ndarray[dtype=np.int8_t, ndim=1] gamma=None):
    cdef int d = Beta.shape[0]
    cdef int i
    cdef double logp = 0.0
    cdef double sum
    if not u is None:
        gamma = np.empty(d, dtype=np.int8)

    for i in xrange(0, d):
        # Compute log conditional probability that gamma(i) is one
        sum = Beta[i, i]
        for j in xrange(i):
            sum += Beta[i, j] * gamma[j]
        logcprob = -log(1 + exp(-sum))

        # Generate the ith entry
        if u is not None: gamma[i] = log(u[i]) < logcprob

        # Add to log conditional probability
        logp += logcprob
        if not gamma[i]: logp -= sum

    return np.array(gamma, dtype=bool), logp