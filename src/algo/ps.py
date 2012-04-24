#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Particle system.
    @namespace algo.ps
    @details
"""

from binary.base import state_space
from binary.product_constrained import ConstrProductBinary
from binary.product_limited import LimitedBinary
from utils.auxi import progress
import numpy
import resample
import sys
import time

class ParticleSystem(object):

    CONST_PRECISION = 1e-8

    def __init__(self, param, job_server=None):
        """
            Constructor.
            \param param parameters
            \param verbose verbose
        """

        ## target function
        self.f = param['f']

        ## dimension of target function
        self.d = self.f.d

        ## number of particles
        if param['smc/n_particles'] is None:
            self.n = int((1.0 - numpy.exp(-self.d / 400.0)) * 25000)
        else:
            self.n = int(param['smc/n_particles'])

        ## verbose outputs
        self.verbose = param['run/verbose']

        ## job server
        self.job_server = job_server

        ## init proposal model
        self.prop = param['smc/binary_model'].uniform(self.d)

        ## array of log weights
        self.log_weights = numpy.zeros(self.n, dtype=float)

        ## array of log evaluation of the proposal model
        self.log_prop = numpy.empty(self.n, dtype=float)

        ## annealing parameter
        self.rho = 0.0

        ## move step counter
        self.n_moves = 0

        ## target function evaluation counter
        self.n_f_evals = 0

        ## acceptance rates
        self.r_ac = list()

        ## particle diversities
        self.r_pd = list()

        self.r_rs = list()

        ## minimum distance of the marginal probability from the boundaries of the unit interval.
        self.eps = param['smc/eps']

        ## minimum correlation required for inclusion in link-conditionals family.
        self.delta = param['smc/delta']

        ## efficient sample size targeted when computing the step length.
        self.eta = param['smc/eta']

        ## powers of two
        self.POWERS_OF_TWO = numpy.array([2 ** i for i in xrange(self.d)])

    def initialize(self, param, target):
        """
            Initialize particle system.
        """
        if self.verbose:
            sys.stdout.write('initialize particle system...')
            t = time.time()

        # sample particles
        if 'prior/model_maxsize' in param.keys() and param['prior/model_maxsize'] is not None:
            u = LimitedBinary(d=self.d, q=param['prior/model_maxsize'])
            self.X = u.rvs(self.n, self.job_server)
        elif 'data/constraints' in param.keys() and param['data/constraints'].shape[0] > 0:
            u = ConstrProductBinary(d=self.d, constrained=param['data/constraints'])
            self.X = u.rvs(self.n, self.job_server)
        else:
            self.X = self.prop.rvs(self.n, self.job_server)

        # compute log(f)
        self.log_f = self.f.lpmf(self.X, self.job_server)
        self.id = numpy.dot(numpy.array(self.X, dtype=int), self.POWERS_OF_TWO)

        # compute first step
        self.reweight(target)

        if self.verbose:
            sys.stdout.write('\rinitialized in %.2f sec\n' % (time.time() - t))

    def enumerate_state_space(self, target):
        """
            Enumerate state space.
        """
        if self.verbose:
            t = time.time()
            sys.stdout.write('enumerate state space...')
        self.X = state_space(self.d)
        self.log_f = self.f.lpmf(self.X, self.job_server)
        self.log_weights = self.log_f * target
        self.rho = target
        if self.verbose:
            sys.stdout.write('\rstate space enumerated in %.2f sec\n' % (time.time() - t))

    def __str__(self):
        """
            \return A string containing the mean of the particle system.
        """
        return '[' + ', '.join(['%.3f' % x for x in self.get_mean()]) + ']'

    def get_mean(self):
        """
            \return Mean of the particle system.
        """
        return numpy.dot(self.get_normalized_weights(), self.X)

    def get_max(self):
        """
            Get maximum and maximizer of particle system.
            \return maximum and maximizer
        """
        i = numpy.argmax(self.log_f)
        return self.log_f[i], self.X[i]

    def get_normalized_weights(self):
        """
            Get normalized weights.
            \return normalized weights
        """
        w = numpy.exp(self.log_weights - max(self.log_weights))
        return w / w.sum()

    def get_id(self, x):
        """
            Get unique ids.
            \param x binary vector.
            \return Unique id.
        """
        return numpy.dot(self.POWERS_OF_TWO, numpy.array(x, dtype=int))

    def get_ess(self, alpha=None):
        """ 
            Computes the effective sample size (ess).
            \param alpha advance of the geometric bridge
            \return ess
        """
        if alpha is None: w = self.log_weights
        else:             w = self.log_weights + alpha * self.log_f
        w = numpy.exp(w - w.max())
        w /= w.sum()
        return 1 / (pow(w, 2).sum())

    def get_particle_diversity(self):
        """
            Computes the particle diversity.
            \return particle diversity
        """
        return numpy.unique(self.id).shape[0] / float(self.n)

    def get_distinct(self):
        """
            Get system of distinct particles with aggregated weights
            \return X, weights
        """
        weights = self.get_normalized_weights()

        # order the data array
        lexorder = numpy.lexsort(numpy.array(self.X).T)

        # check if all entries are equal
        if weights[lexorder[0]] == weights[lexorder[-1]]:
            _X, _weights = numpy.array([self.X[0]]), numpy.array([1.0])
        else:
            _X, _weights = list(), list()

            # loop over ordered data
            x, w = self.X[lexorder[0]], weights[lexorder[0]]

            count = 1
            for i in numpy.append(lexorder[1:], lexorder[0]):
                if (x == self.X[i]).all():
                    count += 1
                else:
                    _X += [x]
                    _weights += [numpy.log(w * count)]
                    x = self.X[i]
                    w = weights[i]
                    count = 1
        return numpy.array(_X, dtype=float), numpy.array(_weights, dtype=float)

    def reweight(self, target):
        """
            Computes an advance of the geometric bridge such that ESS = eta and
            updates the log weights.q. The geometric bridge ends at the target
            distribution.
        """

        l = 0.0; u = target + 0.05 - self.rho
        alpha = min(0.05, u)

        ess = self.get_ess()
        eta = self.eta * ess

        # run bi-sectional search
        for i in xrange(30):
            if self.get_ess(alpha) < eta:
                u = alpha; alpha = 0.5 * (alpha + l)
            else:
                l = alpha; alpha = 0.5 * (alpha + u)

            if abs(l - u) < self.CONST_PRECISION or self.rho + l > 1.0: break

        # update rho and and log weights
        if self.rho + alpha > target:
            alpha = target - self.rho

        # print progress bar
        if not self.verbose:
            progress(ratio=self.rho + alpha, last_ratio=self.rho)
        self.rho += alpha
        self.log_weights += alpha * self.log_f

    def fit_proposal(self):
        """
            Adjust the proposal model to the particle system.
        """
        if self.verbose:
            sys.stdout.write('fitting proposal...')
            t = time.time()

        X, weights = self.get_distinct()
        self.prop.renew_from_data(X=X, weights=weights,
                                  job_server=self.job_server,
                                  eps=self.eps,
                                  delta=self.delta)
        if self.verbose:
            sys.stdout.write('\rfitted proposal in %.2f sec\n' % (time.time() - t))

    def resample(self):
        """
            Resamples the particle system.
        """

        if self.verbose:
            t = time.time()
            sys.stdout.write('resampling...')

        # resample indices
        indices = resample.resample_systematic(self.get_normalized_weights(), numpy.random.random())

        # arrange objects according to resampled order
        self.id = numpy.array([self.id[i] for i in indices])
        self.X = self.X[indices]
        self.log_f = self.log_f[indices]
        self.log_weights = numpy.zeros(self.n)

        # update log proposal values after fitting of proposal
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
            pD = self.get_particle_diversity()
            sys.stdout.write('\rresampled in %.2f sec, pD: %.3f\n' % (time.time() - t, pD))

    def move(self):
        """ 
            Moves the particle system according to an independent Metropolis-
            Hastings kernel to fight depletion of the particle system.
        """

        prev_pD = 0
        for i in xrange(10):
            self.n_moves += 1
            accept = self.kernel()
            pD = self.get_particle_diversity()
            self.r_ac += [float(accept) / float(self.n)]
            self.r_pd += [pD]
            if pD - prev_pD < 0.04 or pD > 0.93: break
            else: prev_pD = pD
        self.r_rs += [i + 1]

    def kernel(self):
        """
            Propagates the particle system via an independent Metropolis Hasting
            kernel.
        """

        self.n_f_evals += self.n

        # sample
        if self.verbose:
            sys.stdout.write('sampling...')
            t = time.time()
        Y, log_prop_Y = self.prop.rvslpmf(self.n, self.job_server)
        if self.verbose:
            sys.stdout.write('\rsampled in %.2f sec\n' % (time.time() - t))

        # evaluate
        if self.verbose:
            sys.stdout.write('evaluating...')
            t = time.time()
        log_f_Y = self.f.lpmf(Y, self.job_server)
        if self.verbose:
            sys.stdout.write('\revaluated in %.2f sec\n' % (time.time() - t))

        # move
        if self.verbose:
            sys.stdout.write('moving...')
            t = time.time()
        log_pi_Y = self.rho * log_f_Y
        log_pi_X = self.rho * self.log_f
        log_prop_X = self.log_prop

        accept = numpy.random.random(self.n) < numpy.exp(log_pi_Y - log_pi_X + log_prop_X - log_prop_Y)
        self.X[accept] = Y[accept]
        self.log_f[accept] = log_f_Y[accept]
        self.log_prop[accept] = log_prop_Y[accept]
        for i in xrange(self.n):
            if accept[i]:
                self.id[i] = self.get_id(Y[i])
        if self.verbose:
            sys.stdout.write('\rmoved in %.2f sec\n' % (time.time() - t))
        return accept.sum()
