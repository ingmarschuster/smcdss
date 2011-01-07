#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from binary import *
from csv import reader

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

        # sample
        Y = sample[:, 0]
        X = sample[:, 1:]

        ## Hierachical Bayesian (hb) or Bayesian Information Criterion (bic)
        self.posterior_type = posterior_type
        ## sample size
        self.n = X.shape[0]
        ## X^t times y
        self.XtY = dot(X.T, Y)
        ## X^t times X
        self.XtX = dot(X.T, X)
        ## Y^t times Y
        YtY = dot(Y.T, Y)

        #-------------------------------------------------- Hierachical Bayesian
        if self.posterior_type == 'hb':

            # full linear model
            beta = solve(self.XtX + 1e-6 * eye(self.d), self.XtY, sym_pos=True)
            # variance of full linear model
            sigma2_full_LM = (YtY - dot(self.XtY, beta)) / float(self.n)

            # prior variance of beta
            self.v = 10.0 / sigma2_full_LM

            # inverse gamma prior of sigma^2
            lambda_ = sigma2_full_LM
            # choose nu such that the prior mean is 2 * sigma2_full_LM
            nu_ = 4.0

            ## constant 1
            self.c1 = 0.5 * log(self.v)
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
            self.c3 = 0.5 * log(self.n)


    def hb(self, gamma):
        '''
        Evaluates the log posterior density from a conjugate hierarchical setup ([George, McCulloch 1997], simplified).
        
            gamma    a binary vector
        '''
        d = gamma.sum()
        if d == 0:
            return - inf
        else:
            K = cholesky(self.XtX[gamma, :][:, gamma] + 1 / self.v * eye(d) + 1e-6 * eye(d))
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

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.XtX.shape[0]

    def _lpmf(self, gamma):
        '''
            Unnormalized log-probability mass function.
            @param gamma binary vector
        '''
        if self.posterior_type == 'bic':
            return float(self.bic(gamma))
        else:
            return float(self.hb(gamma))

    def _pmf(self, gamma):
        '''
            Unnormalized probability mass function.
            @param gamma binary vector
        '''
        if not hasattr(self, 'loglevel'): self._explore()
        return exp(self.lpmf(gamma) - self.loglevel)

    def _rvs(self):
        '''
            Generates a sample from the posterior density rejecting from a proposals of independent 1/2-binary variables.   
        '''
        while True:
            proposal = 0.5 * ones(self.d) > rand(self.d)
            if rand() < self.pmf(proposal): return proposal

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

    d = property(fget=getD, doc="dimension")
