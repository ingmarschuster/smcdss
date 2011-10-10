#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Binary model with independent components.
"""

"""
@namespace binary.product_model
$Author: christian.a.schafer@gmail.com $
$Rev: 149 $
$Date: 2011-09-16 17:40:26 +0200 (Fr, 16 Sep 2011) $
@details
"""

from binary_model import *

class PosProductBinary(ProductBinary):
    """ Non-zero binary model with independent components. """

    name = 'positive product family'

    def __init__(self, p=None, name=name, longname='Non-zero binary model with independent components.'):
        """ 
            Constructor.
            @param p mean vector
            @param name name
            @param longname longname
        """
        ProductBinary.__init__(self, name=name, longname=longname)
        if not p is None:
            if isinstance(p, (numpy.ndarray, list)):
                p = numpy.array(p, dtype=float)
            else:
                p = numpy.array([p])     

        logc = numpy.log(1.0 - numpy.cumprod(1.0 - p[::-1])[::-1]) # prob of gamma > 0 at component 1,..,d
        logc = numpy.append(logc, -numpy.inf)

        self.f_lpmf = _lpmf
        self.f_rvs = _rvs
        self.f_rvslpmf = _rvslpmf
        self.param = dict(logp=numpy.log(p), logq=numpy.log(1 - p), logc=logc)

    def __str__(self):
        return utils.format.format_vector(self.p, 'p') + utils.format.format_vector(self.mean, 'mean')

    def getP(self):
        return numpy.exp(self.param['logp'])

    def getMean(self):
        return numpy.exp(self.param['logp'] - self.param['c'][0])

    p = property(fget=getP, doc="p")
    mean = property(fget=getMean, doc="mean")

def _pmf(gamma, param):
    """ 
        Probability mass function.
        @param gamma binary vector
        @param param parameters
        @return log-probabilities
    """
    p, c = numpy.exp(param['logp']), numpy.exp(param['logc'])
    L = numpy.empty(gamma.shape[0])
    for k in xrange(gamma.shape[0]):
        prob = 1.0
        for i, m in enumerate(p):
            if gamma[k, i]: prob *= m
            else: prob *= (1 - m)
            if not gamma[k, :i].any():
                prob /= c[i]
                if not gamma[k, i]: prob *= c[i + 1]
        L[k] = prob
    return L

def _lpmf(gamma, param):
    """ 
        Log-probability mass function.
        @param gamma binary vector
        @param param parameters
        @return log-probabilities
    """
    return _posproduct_all(param, gamma=gamma)[1]

def _rvs(U, param):
    """ 
        Generates a random variable.
        @param U uniform variables
        @param param parameters
        @return binary variables
    """
    return _posproduct_all(param, U=U)[0]

def _rvslpmf(U, param):
    """ 
        Generates a random variable and computes its probability.
        @param U uniform variables
        @param param parameters
        @return binary variables, log-probabilities
    """
    return _posproduct_all(param, U=U)

def _posproduct_all(param, U=None, gamma=None):
    """ 
        Generates a random variable.
        @param U uniform variables
        @param param parameters
        @return binary variables
    """
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

def main():
    pass

if __name__ == "__main__":
    main()
