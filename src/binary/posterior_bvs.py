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
from csv import reader


from binary import ProductBinary

class PosteriorBinary(ProductBinary):
    '''
        Reads a dataset and construct the posterior probabilities of all linear models
        with variables regressed on the first column.
    '''
    def __init__(self, sample, posterior_type='hb'):
        '''
            Constructor.
            @param data_file data file
            @param posterior_type Hierachical Bayesian (hb) or Bayesian Information Criterion (bic)
        '''

        ProductBinary.__init__(self, name='posterior-binary', longname='A posterior distribution of a Bayesian variable selection problem.')

        ## Hierachical Bayesian (hb) or Bayesian Information Criterion (bic)
        self.posterior_type = posterior_type

        # sample
        Y = sample[:, 0]
        X = sample[:, 1:]

        ## sample size
        self.n = X.shape[0]
        ## X^t times y
        self.XtY = numpy.dot(X.T, Y)
        ## X^t times X
        self.XtX = numpy.dot(X.T, X)
        ## Y^t times Y
        YtY = numpy.dot(Y.T, Y)

        #-------------------------------------------------- Hierachical Bayesian
        if self.posterior_type == 'hb':

            # function
            self.f_lpmf = multi_lpmf_hb
            # full linear model
            beta = scipy.linalg.solve(self.XtX + 1e-6 * numpy.eye(self.d), self.XtY, sym_pos=True)
            # variance of full linear model
            sigma2_full_LM = (YtY - numpy.dot(self.XtY, beta)) / float(self.n)
            # prior variance of beta
            self.v = sigma2_full_LM / 10.0 + 1e-8
            # inverse gamma prior of sigma^2
            lambda_ = sigma2_full_LM
            # choose nu such that the prior mean is 2 * sigma2_full_LM
            nu_ = 4.0

            ## constant 1
            self.c1 = 0.5 * numpy.log(1 / self.v)
            ## constant 2
            self.c2 = 0.5 * (self.n + nu_)
            ## constant 3
            self.c3 = nu_ * lambda_ + YtY


        #---------------------------------------- Bayesian Information Criterion
        if self.posterior_type == 'bic':

            ## constant 1
            self.c1 = YtY / float(self.n)
            ## constant 2
            self.c2 = self.XtY / float(self.n)
            ## constant 3
            self.c3 = 0.5 * numpy.log(self.n)


    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.XtX.shape[0]


    def lpmf(self, gamma, job_server=None, verbose=False):
        '''
            Unnormalized log-probability mass function.
            @param gamma binary vector
        '''
        if len(gamma.shape) == 1:
            return self.f_lpmf(gamma, self.XtX, self.XtY, self.v, self.c1, self.c2, self.c3)[0]

        # number of models
        n = gamma.shape[0]

        # start job server
        if job_server is None:
            arr_lpmf = multi_lpmf_hb(gamma, self.XtX, self.XtY, self.v, self.c1, self.c2, self.c3)
        else:
            ncpus = job_server.get_ncpus()
            jobs = list()
            if verbose: print 'starting %i subprocesses' % ncpus
            for i in xrange(ncpus):
                part = (i * n // ncpus, min((i + 1) * n // ncpus + 1, n))
                jobs.append(
                job_server.submit(
                    func=self.f_lpmf,
                    args=(gamma[part[0]:part[1]], self.XtX, self.XtY, self.v, self.c1, self.c2, self.c3),
                    modules=('numpy', 'scipy.linalg'),
                    )
                )
            # wait for jobs to finish
            job_server.wait()

            # retrieve results
            arr_lpmf = numpy.empty(n, dtype=float)
            for i in xrange(ncpus):
                part = (i * n // ncpus, min((i + 1) * n // ncpus + 1, n))
                arr_lpmf[part[0]:part[1]] = jobs[i]()

        return arr_lpmf

    def _explore(self):
        '''
            Find the maximmum of the log-posterior.
        '''
        ## level of logarithm
        self.loglevel = -inf
        for dec in range(2 ** self.d):
            bin = dec2bin(dec, self.d)
            eval = self.lpmf(bin)
            if eval > self.loglevel: self.loglevel = eval

    def _pmf(self, gamma):
        '''
            Unnormalized probability mass function.
            @param gamma binary vector
        '''
        if not hasattr(self, 'loglevel'): self._explore()
        return exp(self.lpmf(gamma) - self.loglevel)

    d = property(fget=getD, doc="dimension")



def multi_lpmf_hb(gamma, XtX, XtY, v, c1, c2, c3):

    if len(gamma.shape) == 1: gamma = gamma[numpy.newaxis, :]

    # number of models
    n = gamma.shape[0]
    # array to store results
    arr_lpmf = numpy.empty(n, dtype=float)

    for index in xrange(n):
        # model dimension
        d = gamma[index].sum()

        if d == 0:
            # degenerated model
            arr_lpmf[index] = -numpy.inf
        else:
            # regular model
            C = scipy.linalg.cholesky(XtX[gamma[index], :][:, gamma[index]] + v * numpy.eye(d))
            if C.shape == (1, 1): b = XtY[gamma[index], :] / float(C)
            else:                 b = scipy.linalg.solve(C.T, XtY[gamma[index], :])
            k = numpy.log(C.diagonal()).sum()
            arr_lpmf[index] = -k - c1 * d - c2 * numpy.log(c3 - numpy.dot(b, b.T))

    return arr_lpmf



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

