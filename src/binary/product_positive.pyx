#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with non-zero support. \namespace binary.product_positive """

import numpy
cimport numpy

import product
import binary.wrapper as wrapper

class PositiveBinary(product.ProductBinary):
    """ Binary parametric family with non-zero support."""

    name = 'positive product family'

    def __init__(self, p, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param p mean vector
            \param name name
            \param long_name long name
        """

        # call super constructor
        super(PositiveBinary, self).__init__(p=p, name=name, long_name=long_name)

        # add dependent functions
        self.py_wrapper = wrapper.product_positive()
        self.pp_modules += ('binary.product_positive',)

        log_c = numpy.log(1.0 - numpy.cumprod(1.0 - p[::-1])[::-1]) # prob of gamma > 0 at component 1,..,d
        log_c = numpy.append(log_c, -numpy.inf)

        self.log_p = numpy.log(self.p)
        self.log_q = numpy.log(1.0 - self.p)
        self.log_c = log_c

    def __str__(self):
        return 'd: %d, p:\n%smean:\n%s' % (self.d, repr(self.p), repr(self.mean))

    @classmethod
    def _rvslpmf_all(cls, log_p, log_q, log_c, U=None, Y=None):
        """ 
            All-purpose routine for sampling and point-wise evaluation.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        cdef Py_ssize_t size, d, k, i
        
        if U is not None:
            size, d = U.shape
            Y = numpy.empty((size, d), dtype=bool)
            logU = numpy.log(U)

        if Y is not None:
            size, d = Y.shape

        L = numpy.zeros(size, dtype=float)

        for k in xrange(size):

            for i in xrange(d):
                # Compute log conditional probability that gamma(i) is one
                logcprob = log_p[i]
                if not Y[k, :i].any(): logcprob -= log_c[i]

                # Generate the ith entry
                if U is not None: Y[k, i] = logU[k, i] < logcprob

                # Add to log conditional probability
                if Y[k, i]:
                    L[k] += logcprob
                else:
                    L[k] += log_q[i]
                    if not Y[k, :i].any(): L[k] += log_c[i + 1] - log_c[i]

        return Y, L

    @classmethod
    def random(cls, d):
        """ 
            Construct a random product model for testing.
            \param cls class
            \param d dimension
        """
        return cls(0.01 + numpy.random.random(d) * 0.1)

    def _getMean(self):
        return numpy.exp(self.log_p - self.log_c[0])
