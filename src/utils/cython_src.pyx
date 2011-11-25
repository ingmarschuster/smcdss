#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Cython source.
"""

"""
@namespace utils.cython_src
$Author$
$Rev$
$Date$
@details
"""

import numpy
cimport numpy

cdef extern from "math.h":
    double exp(double)
    double log(double)

def resample(numpy.ndarray[dtype=numpy.float64_t, ndim=1] w, double u):
    """
        Computes the particle indices by systematic resampling.
        @param w array of weights
    """
    w = w * w.shape[0]
    cdef int j = 0
    cdef int k
    cdef double cumsum = w[0]
    cdef numpy.ndarray[dtype = numpy.int_t, ndim = 1] i = numpy.zeros(w.shape[0], dtype=numpy.int)
    for k in xrange(w.shape[0]):
        while cumsum < u:
            j = j + 1
            cumsum += w[j]
        i[k] = j
        u = u + 1.0
    return i


def _uniform_all(int q,
                 numpy.ndarray[dtype=numpy.float64_t, ndim=2] U=None,
                 numpy.ndarray[dtype=numpy.int8_t, ndim=2] gamma=None):
    """ Generates a random variable. """
    cdef int k, i, j

    if U is not None:
        size = U.shape[0]
        gamma = numpy.zeros((size, U.shape[1]), dtype=numpy.int8)

    if gamma is not None:
        size = gamma.shape[0]

    cdef int d = gamma.shape[1]

    for k in xrange(size):
        perm = numpy.arange(d, dtype=numpy.int32)
        for i in xrange(d):
            # pick an element in p[:i+1] with which to exchange p[i]
            j = int(U[k, i] * (d - i))
            perm[d - 1 - i], perm[j] = perm[j], perm[d - 1 - i]
        # draw the number of nonzero elements
        r = int(U[k, d - 1] * (q + 1))
        gamma[k, perm[:r]] = True

    return numpy.array(gamma, dtype=bool), -q * log(2) * numpy.ones(d, dtype=float)


def _logistic_all(numpy.ndarray[dtype=numpy.float64_t, ndim=2] Beta,
                  numpy.ndarray[dtype=numpy.float64_t, ndim=2] U=None,
                  numpy.ndarray[dtype=numpy.int8_t, ndim=2] gamma=None):
    """ Generates a random variable. """
    cdef int d = Beta.shape[0]
    cdef int k, i, size
    cdef double logp = 0.0
    cdef double sum

    if U is not None:
        size = U.shape[0]
        gamma = numpy.empty((size, U.shape[1]), dtype=numpy.int8)

    if gamma is not None:
        size = gamma.shape[0]

    L = numpy.zeros(size, dtype=numpy.float64)

    for k in xrange(size):

        for i in xrange(0, d):
            # Compute log conditional probability that gamma(i) is one
            sum = Beta[i, i]
            for j in xrange(i):
                sum += Beta[i, j] * gamma[k, j]
            logcprob = -log(1 + exp(-sum))

            # Generate the ith entry
            if U is not None: gamma[k, i] = log(U[k, i]) < logcprob

            # Add to log conditional probability
            L[k] += logcprob
            if not gamma[k, i]: L[k] -= sum

    return numpy.array(gamma, dtype=bool), L
