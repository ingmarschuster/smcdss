#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Markov chain Monte Carlo on binary spaces.
"""

"""
@namespace ibs.mcmc
$Author$
$Rev$
$Date$
@details The algorithms include the classic Gibbs kernel, Symmetric Metropolis-
Hasting kernels and Adaptvie Metropolis-Hastings kernels.
"""

import time, sys
import numpy
import scipy
import scipy.stats as stats
import utils

class mcmc():
    """ Auxiliary class. """
    header = ['LENGTH', 'NO_EVALS', 'NO_MOVES', 'ACC_RATE' , 'TIME']
    @staticmethod
    def run(v):
        return integrate_mcmc(v)

def integrate_mcmc(v):
    """
        Compute an estimate of the expected value via MCMC.
        @param v parameters
        @param verbose verbose
    """

    mc = MarkovChain(f=v['f'], kernel=v['MCMC_KERNEL'], q=v['MCMC_Q'], max_evals=v['MCMC_MAX_EVALS'], verbose=v['RUN_VERBOSE'])
    mc.burn_in()

    # run for maximum time or maximum iterations
    while not mc.done:
        mc.do_step()
        mc.kernel.adapt(mc.mean, mc.cov)

    print '\nDone.'
    return mc.getCsv()

class MarkovChain():
    """ Markov chain. """

    def __init__(self, f, kernel, q, max_evals=2e6, step_size=1e5, verbose=True):
        """
            Constructor.
            @param f probability mass function 
            @param kernel Markov kernel
            @param q expected number of bits to be flipped
            @param max_evals maximum number of target evaluations
            @param step_size number of steps stored before updating the estimator
            @param verbose verbose
        """

        ## time
        self.t = time.clock()
        ## verbose
        self.verbose = verbose
        ## Markov kernel
        self.kernel = kernel.setup(f, q)
        ## dimension
        self.d = f.d
        ## number of steps stored before updating the estimator
        self.step_size = int(min(step_size, max_evals))
        ## maximum number of target evaluations
        self.max_evals = float(max_evals)
        ## current state
        self.x = numpy.random.random(self.d) > 0.5
        ## log probability of the current state
        self.log_f_x = self.kernel.f.lpmf(self.x)

        ## number of moves
        self.n_moves = 0
        ## number of target function evaluations
        self.n_evals = 0
        ## number of steps
        self.n_steps = 0

        ## mean estimator
        self.mean = numpy.zeros(self.d)
        ## covariance estimator
        self.cov = numpy.zeros((self.d, self.d))

    def __str__(self):
        return '\nmean: %s\nprogress: %.3f %%\nlength: %.3f\nacc_rate: %.3f\nmoves: %.3f\nevals: %.3f' % \
                ('[' + ', '.join(['%.3f' % x for x in self.mean]) + ']',
                 self.progress,
                 self.length * 1e-3,
                 self.acc_rate,
                 self.n_moves * 1e-3,
                 self.n_evals * 1e-3)

    def burn_in(self, n_burn_in=None):
        """
            Run a burn-in period to come closer to the invariant distribution.
            @param n_burn_in number of steps to burn-in 
        """

        if n_burn_in is None: n_burn_in = int(self.max_evals / 100.0)
        last_ratio = 0
        print "\n%s: %i steps burn in..." % (self.kernel.name, n_burn_in)
        for i in xrange(n_burn_in):
            self.x, self.log_f_x, move, eval = self.kernel.rvs(self.x, self.log_f_x)
            last_ratio = utils.format.progress(i / float(n_burn_in), last_ratio=last_ratio)
        if not self.verbose: print "\n%s: %i steps MCMC..." % (self.kernel.name, self.max_evals)

    def do_step(self):
        """ Propagate the Markov chain. """
        mean = numpy.zeros(self.d)
        cov = numpy.zeros((self.d, self.d))
        t = 1.0
        last_ratio = self.n_evals / self.max_evals

        for i in xrange(self.step_size):

            x, self.log_f_x, move, eval = self.kernel.rvs(self.x, self.log_f_x)
            self.n_evals += eval

            if move:
                mean += t * self.x
                cov += t * numpy.dot(x[:, numpy.newaxis], x[ numpy.newaxis, :])
                self.n_moves += 1
                self.x = x
                t = 1.0
            else:
                t += 1.0
            if not self.verbose: last_ratio = utils.format.progress(ratio=self.n_evals / self.max_evals, last_ratio=last_ratio)

        mean /= float(self.step_size)
        cov /= float(self.step_size)
        r = self.n_steps / float(self.n_steps + 1)
        self.mean = r * self.mean + (1 - r) * mean
        self.cov = r * self.cov + (1 - r) * cov
        self.n_steps += 1

        if self.verbose: print self

    def getDone(self):
        return self.max_evals <= self.n_evals

    def getAccRate(self):
        return self.n_moves / float(self.n_evals)

    def getProgress(self):
        return self.n_evals / float(self.max_evals)

    def getLength(self):
        return self.n_steps * self.step_size

    def getCsv(self):
        mean = ','.join(['%.8f' % x for x in self.mean])
        return mean , '%.3f,%.3f,%.3f,%.3f,%.3f' % \
            (self.length * 1e-3,
             self.n_evals * 1e-3,
             self.n_moves * 1e-3,
             self.acc_rate,
             time.clock() - self.t)

    done = property(fget=getDone, doc="is done")
    acc_rate = property(fget=getAccRate, doc="acceptance rate")
    progress = property(fget=getProgress, doc="progress")
    length = property(fget=getLength, doc="length")



class Kernel(stats.rv_discrete):
    """ Wrapper class for Markov kernels. """

    def __init__(self, f, name='Markov kernel', longname='Markov kernel.'):
        """
            Constructor.
            @param f log probability mass function of the invariant distribution 
            @param name name
            @param longname longname
        """
        stats.rv_discrete.__init__(self, name=name, longname=longname)

        ## log probability mass function of the invariant distribution 
        self.f = f
        ## dimension
        self.d = f.d

    @classmethod
    def setup(cls, f, q):
        return cls(f, q)

    def rvs(self, x, log_f_x=None):
        """
            Draw from kernel k(x,\cdot)
            @param x current state
            @param log_f_x log probability of current state
        """
        if log_f_x is None: log_f_x = self.f.lpmf(x)
        return self._rvs(x, log_f_x)

    def adapt(self, mean, cov):
        """ Adapt the kernel. """
        return

    def proposal(self, x, Index):
        Y = x.copy()
        for index in Index:
            Y[index] = Y[index] ^ True
        log_f_Y = self.f.lpmf(Y)
        return Y, log_f_Y

    def getPermutation(self):
        """
            Draw a random permutation of the index set.
            @return permutation
        """
        perm = range(self.d)
        for i in reversed(range(1, self.d)):
            # pick an element in p[:i+1] with which to exchange p[i]
            j = numpy.random.randint(low=0, high=i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm


class Gibbs(Kernel):
        """ Gibbs kernel. """

        def __init__(self, f, q=1.0, name='Gibbs kernel', longname='Gibbs kernel.'):
            """
                Constructor.
                @param f log probability mass function of the invariant distribution 
            """
            Kernel.__init__(self, f, name, longname)
            self.q = q

        def _rvs(self, x, log_f_x):
            """
                Draw from Gibbs kernel k(x,\cdot)
                @param x current state
                @param log_f_x log probability of current state
            """
            Y, log_f_Y = self.proposal(x, Index=[numpy.random.randint(low=0, high=self.d)])

            if numpy.random.random() < 1.0 / (1.0 + numpy.exp(log_f_x - log_f_Y)):
                return Y, log_f_Y, True, True
            else:
                return x, log_f_x, False, True

class SwapMetropolisHastings(Gibbs):
        """ Swap Metropolis-Hastings kernel. """

        def __init__(self, f, q, name='Swap Metropolis-Hastings kernel', longname='Swap Metropolis-Hastings kernel.'):
            """
                Constructor.
                @param f log probability mass function of the invariant distribution 
            """
            Gibbs.__init__(self, f, q, name=name, longname=longname)

        def _rvs(self, x, log_f_x):
            """
                Draw from Swap Metropolis-Hastings kernel k(x,\cdot)
            """

            Y = x.copy()
            if numpy.random.random() < 0.5:
                index = numpy.random.randint(low=0, high=self.d)
                Y[index] = Y[index] ^ True
            else:
                index_in = numpy.where(Y)[0]
                index_out = numpy.where(Y ^ True)[0]
                Y[index_in[numpy.random.randint(low=0, high=index_in.shape[0])]] = False
                Y[index_out[numpy.random.randint(low=0, high=index_out.shape[0])]] = True
            log_f_Y = self.f.lpmf(Y)
            if numpy.random.random() < numpy.exp(log_f_Y - log_f_x):
                return Y, log_f_Y, True, True
            else:
                return x, log_f_x, False, True

class SymmetricMetropolisHastings(Gibbs):
        """ Symmetric Metropolis-Hastings kernel. """

        def __init__(self, f, q, name='Symmetric Metropolis-Hastings kernel', longname='Symmetric Metropolis-Hastings kernel.'):
            """
                Constructor.
                @param f log probability mass function of the invariant distribution 
            """
            Gibbs.__init__(self, f, q, name=name, longname=longname)

        def _rvs(self, x, log_f_x):
            """
                Draw from Symmetric Metropolis-Hastings kernel k(x,\cdot)
            """
            Y, log_f_Y = self.proposal(x, Index=self.getRandomSubset())
            if numpy.random.random() < numpy.exp(log_f_Y - log_f_x):
                return Y, log_f_Y, True, True
            else:
                return x, log_f_x, False, True

        def getRandomSubset(self):
            """
                Draw a uniformly random subset of the index set.
                @return subset
            """
            if self.q == 1: return [numpy.random.randint(low=0, high=self.d)]
            k = min(stats.geom.rvs(1.0 / self.q), self.d)
            if k < 5:
                Index = []
                while len(Index) < k:
                    n = numpy.random.randint(low=0, high=self.d)
                    if not n in Index: Index.append(n)
            else:
                Index = self.getPermutation()[:k]
            return Index


class AdaptiveMetropolisHastings(Kernel):
        """ Adaptive Metropolis-Hastings kernel"""

        def __init__(self, f, q=1.0, name='Adaptive Metropolis-Hastings kernel',
                     longname='Adaptive Metropolis-Hastings kernel.'):
            """
                Constructor.
                @param f log probability mass function of the invariant distribution
                @param q expected number of bits to be flipped
                @param name name
                @param longname longname
            """
            Kernel.__init__(self, f=f, name=name, longname=longname)
            self.adapted = False
            self.mean = 0.5 * numpy.ones(self.d)
            self.W = numpy.eye(self.d)
            self.k = 0
            self.perm = range(self.d)
            self.delta = 0.01
            self.xlambda = 0.01

        def adapt(self, mean, cov):
            """ Adapt the kernel. """
            self.mean = mean
            self.W = scipy.linalg.inv(cov + self.xlambda * numpy.eye(self.d))
            self.adapted = True

        def _rvs(self, x, log_f_x):
            """
                Draw from Metropolised Gibbs kernel k(x,\cdot)
                @param x current state
                @param log_f_x log probability of current state
            """
            if self.k == self.d:
                self.perm = self.getPermutation()
                self.k = 0
            j = self.perm[self.k]
            self.k += 1

            if self.adapted:
                not_j = [i for i in self.perm if not i == j]
                v = numpy.dot(self.W[j, not_j], x[not_j] - self.mean[not_j])
                psi = self.mean[j] - v / self.W[j, j]

                q = max(min(psi, 1 - self.delta), self.delta)

                # return if there is no mutation
                if (numpy.random.random() < q) == x[j]: return x, log_f_x, False, False
            else:
                q = 0.5

            Y, log_f_Y = self.proposal(x, Index=[j])

            if Y[j]:
                r = (1 - q) / q
            else:
                r = q / (1 - q)

            if numpy.random.random() < numpy.exp(log_f_Y - log_f_x) * r:
                return Y, log_f_Y, True, True
            else:
                return x, log_f_x, False, True
