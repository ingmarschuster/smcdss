#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#    $Author: Christian Schäfer
#    $Date$

__version__ = "$Revision$"

import time, datetime, sys, operator
import numpy

import ibs
import utils

class smc():
    header = ['NO_EVALS', 'TIME']
    @staticmethod
    def run(v):
        return integrate_smc(v)

def integrate_smc(param):

    sys.stdout.write('running smc')

    ps = ParticleSystem(param)

    # run sequential MC scheme
    while ps.rho < 1.0:

        ps.fit_proposal()
        ps.resample()
        ps.move()
        ps.reweight()

    sys.stdout.write('\rsmc completed in %s.\n' % (str(datetime.timedelta(seconds=time.time() - ps.start))))

    return ps.getCsv()


class ParticleSystem(object):

    def __init__(self, v):
        '''
            Constructor.
            @param param parameters
            @param verbose verbose
        '''
        self.verbose = v['RUN_VERBOSE']
        if self.verbose: sys.stdout.write('...\n\n')

        self.start = time.time()

        if 'cython' in utils.opts: self._resample = utils.cython.resample
        else: self._resample = utils.python.resample

        ## target function
        self.f = v['f']
        self.job_server = v['JOB_SERVER']

        ## proposal model
        self.prop = v['SMC_BINARY_MODEL'].uniform(self.f.d)

        ## dimension of target function
        self.d = self.f.d
        ## number of particles
        self.n = v['SMC_N_PARTICLES']

        ## array of log weights
        self.log_W = numpy.zeros(self.n, dtype=float)
        ## array of log evaluation of the proposal model
        self.log_prop = numpy.empty(self.n, dtype=float)
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
        self.eps = v['SMC_EPS']
        ## min correlation to be considered part of a logistic model
        self.delta = v['SMC_DELTA']

        self.__k = numpy.array([2 ** i for i in xrange(self.d)])

        if self.verbose:
            sys.stdout.write('initializing...')
            t = time.time()
        self.X = self.prop.rvs(self.n, self.job_server)
        self.log_f = self.f.lpmf(self.X, self.job_server)
        for i in xrange(self.n): self.id[i] = self.getId(self.X[i])
        if self.verbose: print '\rinitialized in %.2f sec' % (time.time() - t)

        # do first step
        self.reweight()

    def __str__(self):
        return '[' + ', '.join(['%.3f' % x for x in self.getMean()]) + ']'

    def getCsv(self):
        return (','.join(['%.8f' % x for x in self.getMean()]),
                ','.join(['%.3f' % (self.n_f_evals / 1000.0), '%.3f' % (time.time() - self.start)]),
                ','.join(['%.5f' % x for x in self.r_pd]),
                ','.join(['%.5f' % x for x in self.r_ac]),
                ','.join(['%.5f' % x for x in self.log_f]))

    def getMean(self):
        return numpy.dot(self.nW, self.X)

    def getId(self, x):
        '''
            Assigns a unique id to x.
            @param x binary vector.
            @return id
        '''
        return numpy.dot(self.__k, numpy.array(x, dtype=int))

    def getEss(self, alpha=None):
        ''' Computes the effective sample size (ess).
            @param alpha advance of the geometric bridge
            @return ess
        '''
        if alpha is None: w = self.log_W
        else:             w = alpha * self.log_f
        w = numpy.exp(w - w.max())
        w /= w.sum()
        return 1 / (self.n * pow(w, 2).sum())

    def getParticleDiversity(self):
        ''' Computes the particle diversity.
            @return particle diversity
        '''
        dic = {}
        map(operator.setitem, (dic,)*self.n, self.id, [])
        return len(dic.keys()) / float(self.n)

    def reweight(self):
        ''' Computes an advance of the geometric bridge such that ess = tau and updates the log weights. '''
        l = 0.0; u = 1.05 - self.rho
        alpha = min(0.05, u)

        tau = 0.9

        # run bisectional search
        for iter in xrange(30):

            if self.getEss(alpha) < tau:
                u = alpha; alpha = 0.5 * (alpha + l)
            else:
                l = alpha; alpha = 0.5 * (alpha + u)

            if abs(l - u) < ibs.CONST_PRECISION or self.rho + l > 1.0: break

        # update rho and and log weights
        if self.rho + alpha > 1.0: alpha = 1.0 - self.rho
        self.rho += alpha
        self.log_W = alpha * self.log_f

        if self.verbose:
            utils.format.progress(ratio=self.rho, text='\n')
            print '\n' + str(self) + '\n'

    def fit_proposal(self):
        ''' Adjust the proposal model to the particle system.
            @todo sample.distinct could ba activated for speedup
        '''
        if self.verbose:
            sys.stdout.write('fitting proposal...')
            t = time.time()
        sample = utils.data.data(self.X, self.log_W)
        # sample.distinct()
        self.prop.renew_from_data(sample, job_server=self.job_server, eps=self.eps, delta=self.delta, verbose=False)
        if self.verbose: print '\rfitted proposal in %.2f sec' % (time.time() - t)

    def getNWeight(self):
        '''
            Get the normalized weights.
            @return normalized weights
        '''
        w = numpy.exp(self.log_W - max(self.log_W))
        return w / w.sum()

    def getSystemStructure(self):
        '''
            Gather a summary of how many particles are n-fold in the particle system.
        '''
        id_set = set(self.id)
        l = [ self.id.count(i) for i in id_set ]
        k = [ l.count(i) * i for i in xrange(1, 101) ]
        return str(k) + ' %i ' % sum(k)

    def getMax(self):
        index = numpy.argmax(self.log_f)
        return self.log_f[index], self.X[index]

    def move(self):
        ''' Moves the particle system according to an independent Metropolis-Hastings kernel
            to fight depletion of the particle system.
        '''

        prev_pD = 0
        self.r_ac += [0]
        for iter in xrange(10):
            self.n_moves += 1
            accept = self.kernel()
            self.r_ac[-1] += accept
            pD = self.pD
            if self.verbose: print "moved with aR: %.3f, pD: %.3f" % (accept / float(self.n), pD)
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

        # sample
        if self.verbose:
            sys.stdout.write('sampling...')
            t = time.time()
        Y, log_prop_Y = self.prop.rvslpmf(self.n, self.job_server)
        if self.verbose: print '\rsampled in %.2f sec' % (time.time() - t)

        # evaluate
        if self.verbose:
            sys.stdout.write('evaluating...')
            t = time.time()
        log_f_Y = self.f.lpmf(Y, self.job_server)
        if self.verbose: print '\revaluated in %.2f sec' % (time.time() - t)

        # move
        log_pi_Y = self.rho * log_f_Y
        log_pi_X = self.rho * self.log_f
        log_prop_X = self.log_prop

        accept = numpy.random.random(self.n) < numpy.exp(log_pi_Y - log_pi_X + log_prop_X - log_prop_Y)
        self.X[accept] = Y[accept]
        self.log_f[accept] = log_f_Y[accept]
        self.log_prop[accept] = log_prop_Y[accept]
        for index in xrange(self.n):
            if accept[index]:
                self.id[index] = self.getId(Y[index])
        return accept.sum()

    def resample(self):
        ''' Resamples the particle system. '''

        if self.verbose:
            t = time.time()
            sys.stdout.write('resampling...')
        indices = self._resample(self.nW, numpy.random.random())

        # move objects according to resampled order
        self.id = [self.id[i] for i in indices]
        self.X = self.X[indices]
        self.log_f = self.log_f[indices]

        pD = self.pD

        # update log proposal values
        if not self.job_server is None and self.job_server.get_ncpus() > 1:
            self.log_prop = self.prop.lpmf(self.X, self.job_server)
        else:
            self.log_prop[0] = self.prop.lpmf(self.X[0])
            for i in xrange(1, self.n):
                if (self.log_prop[i] == self.log_prop[i - 1]).all():
                    self.log_prop[i] = self.log_prop[i - 1]
                else:
                    self.log_prop[i] = self.prop.lpmf(self.X[i])

        if self.verbose:
            print '\rresampled in %.2f sec, pD: %.3f' % (time.time() - t, pD)

    nW = property(fget=getNWeight, doc="normalized weights")
    pD = property(fget=getParticleDiversity, doc="particle diversity")
