#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with independent components and positive support."""

"""
\namespace binary.product_pos
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

import cython
import numpy
cimport numpy

import utils
import binary.product
import binary.wrapper

def _posproduct_all(param, U=None, gamma=None):
    """ 
        All-purpose routine for sampling and point-wise evaluation.
        \param U uniform variables
        \param param parameters
        \return binary variables
    """
    cdef int size, d, k, i
    logp, logq, logc = param['logp'], param['logq'], param['logc']
    if U is not None:
        size = U.shape[0]
        d = U.shape[1]
        gamma = numpy.empty((size, U.shape[1]), dtype=bool)
        logU = numpy.log(U)

    if gamma is not None:
        size = gamma.shape[0]
        d = gamma.shape[1]

    L = numpy.zeros(size, dtype=float)

    for k in xrange(size):

        for i in xrange(d):
            # Compute log conditional probability that gamma(i) is one
            logcprob = logp[i]
            if not gamma[k, :i].any(): logcprob -= logc[i]

            # Generate the ith entry
            if U is not None: gamma[k, i] = logU[k, i] < logcprob

            # Add to log conditional probability
            if gamma[k, i]:
                L[k] += logcprob
            else:
                L[k] += logq[i]
                if not gamma[k, :i].any(): L[k] += logc[i + 1] - logc[i]

    return gamma, L


class PosProductBinary(binary.product.ProductBinary):
    """ Binary parametric family with independent components and positive support."""

    def __init__(self, p=None, py_wrapper=None, name='positive product family', long_name=__doc__):
        """ 
            Constructor.
            \param p mean vector
            \param name name
            \param long_name long name
        """

        if cython.compiled: print "Yep, I'm compiled."
        else: print "Just a lowly interpreted script."

        # link to python wrapper
        if py_wrapper is None: py_wrapper = binary.wrapper.pos_product()

        # call super constructor
        binary.product.ProductBinary.__init__(self, py_wrapper=py_wrapper, name=name, long_name=long_name)

        # add modules
        self.pp_modules = ('numpy', 'binary.pos_product',)

        # add dependent functions
        self.pp_depfuncs += ('_posproduct_all',)

        if not p is None:
            if isinstance(p, (numpy.ndarray, list)):
                p = numpy.array(p, dtype=float)
            else:
                p = numpy.array([p])

        logc = numpy.log(1.0 - numpy.cumprod(1.0 - p[::-1])[::-1]) # prob of gamma > 0 at component 1,..,d
        logc = numpy.append(logc, -numpy.inf)

        self.param.update(dict(logp=numpy.log(p), logq=numpy.log(1 - p), logc=logc))

    def __str__(self):
        return utils.format.format_vector(self.p, 'p') + utils.format.format_vector(self.mean, 'mean')

    def _getP(self):
        """ Get p-vector of instance. \return p-vector """
        return numpy.exp(self.param['logp'])

    def _getMean(self):
        return numpy.exp(self.param['logp'] - self.param['logc'][0])

