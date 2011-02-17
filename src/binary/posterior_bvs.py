#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

import scipy.linalg
import numpy
import pp

from binary import ProductBinary

class PosteriorBinary(ProductBinary):
    '''
        Reads a dataset and construct the posterior probabilities of all linear models
        with variables regressed on the first column.
    '''
    def __init__(self, sample, posterior_type='hb'):
        '''
            Constructor.
            @param sample data to perform variable selection on
            @param posterior_type Hierachical Bayesian (hb) or Bayesian Information Criterion (bic)
        '''

        ProductBinary.__init__(self, name='posterior-binary', longname='A posterior distribution of a Bayesian variable selection problem.')

        ## Hierachical Bayesian (hb) or Bayesian Information Criterion (bic)
        self.posterior_type = posterior_type
        # sample
        Y = sample[:, 0]
        X = sample[:, 1:]
        n = X.shape[0]
        d = X.shape[1]
        XtY = numpy.dot(X.T, Y)
        XtX = numpy.dot(X.T, X)
        YtY = numpy.dot(Y.T, Y)

        if self.posterior_type == 'hb':
            ## log posterior function
            self.f_lpmf = _lpmf_hb
            # full linear model
            beta = scipy.linalg.solve(XtX + 1e-6 * numpy.eye(d), XtY, sym_pos=True)
            # variance of full linear model
            sigma2_full_LM = (YtY - numpy.dot(XtY, beta)) / float(n)
            # prior variance of beta
            v = sigma2_full_LM / 10.0 + 1e-8
            # inverse gamma prior of sigma^2
            lambda_ = sigma2_full_LM
            # choose nu such that the prior mean is 2 * sigma2_full_LM
            nu_ = 4.0
            ## constant 1
            c1 = 0.5 * numpy.log(1 / v)
            ## constant 2
            c2 = 0.5 * (n + nu_)
            ## constant 3
            c3 = nu_ * lambda_ + YtY

            self.param = dict(XtX=XtX, XtY=XtY, v=v, c1=c1, c2=c2, c3=c3)

        if self.posterior_type == 'bic':
            ## constant 1
            c1 = YtY / float(n)
            ## constant 2
            c2 = self.XtY / float(n)
            ## constant 3
            c3 = 0.5 * numpy.log(n)

    def getD(self):
        ''' Get dimension.
            @return dimension 
        '''
        return self.param['XtX'].shape[0]

    def __explore(self):
        ''' Find the maximmum of the log-posterior.
            @deprecated method is never used.
        '''
        ## level of logarithm
        self.loglevel = -inf
        for dec in range(2 ** self.d):
            bin = dec2bin(dec, self.d)
            eval = self.lpmf(bin)
            if eval > self.loglevel: self.loglevel = eval

    def __pmf(self, gamma):
        ''' Unnormalized probability mass function.
            @deprecated method is never used.
            @param gamma binary vector
        '''
        if not hasattr(self, 'loglevel'): self._explore()
        return exp(self.lpmf(gamma) - self.loglevel)

    d = property(fget=getD, doc="dimension")


def _lpmf_hb(gamma, param):
    ''' Log-probability mass function.
        @param gamma binary vector
        @param param parameters
        @return log-probabilities
    '''
    XtX = param['XtX']
    XtY = param['XtY']

    # number of models
    size = gamma.shape[0]
    # array to store results
    L = numpy.empty(size, dtype=float)

    for k in xrange(size):
        # model dimension
        d = gamma[k].sum()
        if d == 0:
            # degenerated model
            L[k] = -numpy.inf
        else:
            # regular model
            C = scipy.linalg.cholesky(XtX[gamma[k], :][:, gamma[k]] + param['v'] * numpy.eye(d))
            if C.shape == (1, 1): b = XtY[gamma[k], :] / float(C)
            else:                 b = scipy.linalg.solve(C.T, XtY[gamma[k], :])
            log_diag_C = numpy.log(C.diagonal()).sum()
            L[k] = -log_diag_C - param['c1'] * d - param['c2'] * numpy.log(param['c3'] - numpy.dot(b, b.T))

    return L


"""
    def hb(self, gamma):
        '''
        Evaluates the log posterior density from a conjugate hierarchical setup ([George, McCulloch 1997], simplified).
        
            gamma    a binary vector
        '''
        d = gamma.sum()
        if d == 0:
            return - inf
        else:
            K = cholesky(self.XtX[gamma, :][:, gamma] + self.v * eye(d))
            if K.shape == (1, 1):
                w = self.XtY[gamma, :] / float(K)
            else:
                w = solve(K.T, self.XtY[gamma, :])
            k = log(K.diagonal()).sum()
        return - k - self.c1 * d - self.c2 * log(self.c3 - dot(w, w.T))


    def bic(self, gamma):
        '''
        Evaluates the Bayesian Information Criterion. 
        
            gamma    a binary vector
        '''
        p = gamma.sum()
        if p == 0:
            return - inf
        else:
            try:
                beta = solve(self.XtX[gamma, :][:, gamma], self.XtY[gamma, :], sym_pos=True)
            except LinAlgError:
                beta = solve(self.XtX[gamma, :][:, gamma] + exp(-10) * eye(p), self.XtY[gamma, :], sym_pos=True)
        return - 0.5 * self.n * log(self.c1 - dot(self.c2[gamma], beta)) - self.c3 * p
"""

