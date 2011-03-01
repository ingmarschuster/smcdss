#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import time
import pp

from operator import setitem
from sys import stdout
from numpy import *

import binary
import utils

CONST_PRECISION = 1e-8

header = lambda: ['NO_EVALS', 'TIME']

def run(param, verbose=True):

    print 'running smc...\n'

    ps = ParticleSystem(param)

    # run sequential MC scheme
    while ps.rho < 1.0:

        ps.fit_proposal()
        ps.resample()
        ps.move()
        ps.reweight()

    print "\ndone in %.3f seconds.\n" % (time.time() - ps.start)

    return ps.getCsv()


class ParticleSystem(object):

    def __init__(self, param, verbose=True):
        '''
            Constructor.
            @param param parameters
            @param verbose verbose
        '''
        self.verbose = verbose
        self.start = time.time()

        if 'cython' in utils.opts: self._resample = utils.cython.resample
        else: self._resample = utils.python.resample

        ## target function
        self.f = param['f']
        self.job_server = pp.Server(ncpus=param['smc_ncpus'], ppservers=())

        ## proposal model
        self.prop = param['smc_binary_model'].uniform(self.f.d)

        ## dimension of target function
        self.d = self.f.d
        ## number of particles
        self.n = param['smc_n']

        ## array of particles
        self.X = resize(empty(self.n * self.d, dtype=bool), (self.n, self.d))
        ## array of log weights
        self.log_W = zeros(self.n, dtype=float)
        ## array of log evaluations of f
        self.log_f = empty(self.n, dtype=float)
        ## array of log evaluation of the proposal model
        self.log_prop = empty(self.n, dtype=float)
        ## array of ids
        self.id = [0] * self.n

        ## annealing parameter
        self.rho = 0
        ## move step counter
        self.n_moves = 0
        ## target function evaluation counter
        self.n_f_evals = 0

        ## acceptance rates
        self.r_ac = []
        ## particle diversities
        self.r_pd = []

        ## min mean distance from the boundaries of [0,1] to be considered part of a logistic model
        self.eps = param['smc_eps']
        ## min correlation to be considered part of a logistic model
        self.delta = param['smc_delta']
        ## min mean distance from the boundaries of [0,1] to be considered random
        self.xi = param['smc_xi']

        self.__k = array([2 ** i for i in range(self.d)])

        # initialize particle system
        self.X = self.prop.rvs(self.n, self.job_server)
        self.log_f = self.f.lpmf(self.X, self.job_server)
        for i in xrange(self.n):
            self.id[i] = self.getId(self.X[i])

        # do first step
        self.reweight()

    def __str__(self):
        return '[' + ', '.join(['%.3f' % x for x in self.getMean()]) + ']'

    def getCsv(self):
        return ('\t'.join(['%.8f' % x for x in self.getMean()]),
                '\t'.join(['%.3f' % (self.n_f_evals / 1000.0), '%.3f' % (time.time() - self.start)]),
                ','.join(['%.5f' % x for x in self.r_pd]),
                ','.join(['%.5f' % x for x in self.r_ac]),
                ','.join(['%.5f' % x for x in self.log_f]))

    def getMean(self):
        return dot(self.nW, self.X)

    def getId(self, x):
        '''
            Assigns a unique id to x.
            @param x binary vector.
            @return id
        '''
        return dot(self.__k, array(x, dtype=int))

    def getEss(self, alpha=None):
        ''' Computes the effective sample size (ess).
            @param alpha advance of the geometric bridge
            @return ess
        '''
        if alpha is None: w = self.log_W
        else:             w = alpha * self.log_f
        w = exp(w - w.max())
        w /= w.sum()
        return 1 / (self.n * pow(w, 2).sum())

    def getParticleDiversity(self):
        ''' Computes the particle diversity.
            @return particle diversity
        '''
        dic = {}
        map(setitem, (dic,)*self.n, self.id, [])
        return len(dic.keys()) / float(self.n)

    def reweight(self):
        ''' Computes an advance of the geometric bridge such that ess = tau and updates the log weights.
            @todo this should be replaced by cython code.
        '''
        l = 0.0; u = 1.05 - self.rho
        alpha = min(0.05, u)

        tau = 0.9

        # run bisectional search
        for iter in range(30):

            if self.getEss(alpha) < tau:
                u = alpha; alpha = 0.5 * (alpha + l)
            else:
                l = alpha; alpha = 0.5 * (alpha + u)

            if abs(l - u) < CONST_PRECISION or self.rho + l > 1.0: break

        # update rho and and log weights
        if self.rho + alpha > 1.0: alpha = 1.0 - self.rho
        self.rho += alpha
        self.log_W = alpha * self.log_f

        if self.verbose: print 'progress %.1f' % (100 * self.rho) + '%'
        print '\n' + str(self) + '\n'

    def fit_proposal(self):
        ''' Adjust the proposal model to the particle system.
            @todo sample.distinct could ba activated for speedup
        '''
        sample = utils.data.data(self.X, self.log_W)
        # sample.distinct()
        self.prop.renew_from_data(sample, eps=self.eps, delta=self.delta, xi=self.xi, verbose=self.verbose)

    def getNWeight(self):
        '''
            Get the normalized weights.
            @return normalized weights
        '''
        w = exp(self.log_W - max(self.log_W))
        return w / w.sum()

    def getSystemStructure(self):
        '''
            Gather a summary of how many particles are n-fold in the particle system.
        '''
        id_set = set(self.id)
        l = [ self.id.count(i) for i in id_set ]
        k = [ l.count(i) * i for i in range(1, 101) ]
        return str(k) + ' %i ' % sum(k)

    def move(self):
        ''' Moves the particle system according to an independent Metropolis-Hastings kernel
            to fight depletion of the particle system.
        '''

        prev_pD = 0
        self.r_ac += [0]
        for iter in range(10):

            self.n_moves += 1
            # pass particle system through transition kernel
            n_acceptance = self.kernel()

            pD = self.pD
            if self.verbose: print "\naR: %.3f, pD: %.3f" % (n_acceptance / float(self.n), pD)
            self.r_ac[-1] += n_acceptance

            if pD - prev_pD < 0.04 or pD > 0.93: break
            else: prev_pD = pD

        self.r_ac[-1] /= ((iter + 1) * float(self.n))
        self.r_pd += [pD]

    def kernel(self):
        '''
            Propagates the particle system via an independent Metropolis Hasting kernel.
            @todo do accept/reject step vectorized
        '''

        self.n_f_evals += self.n
        print 'sample proposals...'
        Y, log_prop_Y = self.prop.rvslpmf(self.n, self.job_server)

        print 'evaluate proposals...'
        log_f_Y = self.f.lpmf(Y, self.job_server)

        print 'do MH steps...'
        n_acceptance = 0
        for index in range(self.n):

            # values proposal Y
            log_pi_y = self.rho * log_f_Y[index]
            log_prop_y = log_prop_Y[index]
            # values state X
            log_pi_X = self.rho * self.log_f[index]
            log_prop_X = self.log_prop[index]

            # compute acceptance probability and do MH step
            if random.random() < exp(log_pi_y - log_pi_X + log_prop_X - log_prop_y):
                self.X[index] = Y[index]
                self.id[index] = self.getId(Y[index])
                self.log_f[index] = log_f_Y[index]
                self.log_prop[index] = log_prop_Y[index]
                n_acceptance += 1

        print 'move completed.\n'
        return n_acceptance

    def resample(self):
        ''' Resamples the particle system. '''
        
        if self.verbose: print "resample. ",
        indices = self._resample(self.nW, random.random())

        # move objects according to resampled order
        self.id = [self.id[i] for i in indices]
        self.X = self.X[indices]
        self.log_f = self.log_f[indices]

        pD = self.pD
        if self.verbose: print 'pD: %.3f' % pD

        # update log proposal values
        if pD > 0.5:
            self.log_prop = self.prop.lpmf(self.X, self.job_server)
        else:
            self.log_prop[0] = self.prop.lpmf(self.X[0])
            for i in xrange(1, self.n):
               if (self.log_prop[i] == self.log_prop[i - 1]).all():
                   self.log_prop[i] = self.log_prop[i - 1]
               else:
                   self.log_prop[i] = self.prop.lpmf(self.X[i])

    nW = property(fget=getNWeight, doc="normalized weights")
    pD = property(fget=getParticleDiversity, doc="particle diversity")
