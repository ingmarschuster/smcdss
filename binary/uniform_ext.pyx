#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Cython extension. """

"""
\namespace binary.uniform_ext 
$Author: christian.a.schafer@gmail.com $
$Rev: 165 $
$Date: 2011-11-25 18:05:05 +0100 (ven., 25 nov. 2011) $
"""

import numpy
cimport numpy

def _rvs(int q, numpy.ndarray[dtype=numpy.float64_t, ndim=2] U):
    """
        All-purpose routine for sampling and point-wise evaluation.
        \param U uniform variables
        \param param parameters
        \return binary variables
    """
    cdef int k, i, j
    cdef int size = U.shape[0]
    cdef int d = U.shape[1]
    
    gamma = numpy.zeros((size, d), dtype=numpy.int8)

    for k in xrange(size):
        perm = numpy.arange(d, dtype=numpy.int32)
        for i in xrange(d):
            # pick an element in p[:i+1] with which to exchange p[i]
            j = int(U[k, i] * (d - i))
            perm[d - 1 - i], perm[j] = perm[j], perm[d - 1 - i]
        # draw the number of nonzero elements
        r = int(U[k, d - 1] * (q + 1))
        gamma[k, perm[:r]] = True

    return numpy.array(gamma, dtype=bool)
