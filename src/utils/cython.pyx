#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import numpy as np
cimport numpy as np

def resample(np.ndarray[dtype=np.float64_t, ndim=1] w, double u):
   '''
       Computes the particle indices by systematic resampling.
       @param w array of weights
   '''
   w = w * w.shape[0]
   cdef int j = 0
   cdef int k
   cdef double cumsum = w[0]
   cdef np.ndarray i = np.zeros(w.shape[0], dtype=np.int)
   for k in xrange(w.shape[0]):
       while cumsum < u:
           j = j + 1
           cumsum += w[j]
       i[k] = j
       u = u + 1.0
   return i