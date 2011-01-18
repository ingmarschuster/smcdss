#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-12-02 20:52:23 +0100 (mar., 30 nov. 2010) $
    $Revision: 1 $
'''

from time import clock
from auxpy.data import format
from numpy import *
from auxpy.data import data
from sys import stdout
from scipy.stats import rv_discrete, geom

header = lambda: ['LENGTH', 'NO_EVALS', 'NO_MOVES', 'ACC_RATE' , 'TIME']

def run(param, verbose=True):

    mc = MarkovChain(f=param['f'], kernel=param['mcmc_kernel'], q=param['mcmc_q'], max_evals=param['mcmc_max_evals'])

    mc.burn_in()

    # run for maximum time or maximum iterations
    while not mc.done:
        mc.do_step()
        mc.kernel.adapt(mc.mean, mc.cov)

    return mc.getCsv()


class MarkovChain():

    def __init__(self, f, kernel, q, max_evals=1e6, step_size=1e4, verbose=True):

        self.start = clock()
        self.verbose = verbose
        self.kernel = kernel.setup(f, q)
        self.d = f.d
        self.step_size = int(min(step_size, max_evals))
        self.max_evals = float(max_evals)
        self.x = random.random(self.d) > 0.5
        self.log_f_x = self.kernel.f.lpmf(self.x)

        ## number of moves
        self.n_moves = 0
        ## number of target function evaluations
        self.n_evals = 0
        ## number of steps
        self.n_steps = 0

        self.mean = zeros(self.d)
        self.cov = zeros((self.d, self.d))

    def __str__(self):
        return '\nprogress: %.3f %%\nlength: %.3f\nacc_rate: %.3f\nmoves: %.3f\nevals: %.3f' % \
                (self.progress,
                 self.length * 1e-3,
                 self.acc_rate,
                 self.n_moves * 1e-3,
                 self.n_evals * 1e-3)

    def burn_in(self, n_burn_in=None):

        if n_burn_in is None: n_burn_in = int(self.max_evals / 100.0)

        if self.verbose:
            print "\n%s: %i steps burn in..." % (self.kernel.name, n_burn_in)
            stdout.write("" + 101 * " " + "]" + "\r" + "[")
            self.n_bars = 0

        for i in range(n_burn_in):
            self.x, self.log_f_x, move, eval = self.kernel.rvs(self.x, self.log_f_x)

            # print progress bar
            if self.verbose:
                if self.n_bars < int(100.0 * i / float(n_burn_in)):
                    stdout.write("-")
                    stdout.flush()
                    self.n_bars += 1

        if not self.verbose:
            print "\n%s: %i steps mcmc..." % (self.kernel.name, self.max_evals)
            stdout.write("" + 101 * " " + "]" + "\r" + "[")
            self.n_bars = 0

    def do_step(self):
        mean = zeros(self.d)
        cov = zeros((self.d, self.d))
        t = 1.0

        for i in xrange(self.step_size):

            x, self.log_f_x, move, eval = self.kernel.rvs(self.x, self.log_f_x)
            self.n_evals += eval

            if move:
                mean += t * self.x
                cov += t * dot(x[:, newaxis], x[newaxis, :])
                self.n_moves += 1
                self.x = x
                t = 1.0
            else:
                t += 1.0

            # print progress bar
            if not self.verbose:
                if self.n_bars < int(100.0 * self.n_evals / self.max_evals):
                    stdout.write("-")
                    stdout.flush()
                    if self.n_evals >= self.max_evals: break
                    self.n_bars += 1

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
        mean = '\t'.join(['%.8f' % x for x in self.mean])
        return mean , '%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % \
            (self.length * 1e-3,
             self.n_evals * 1e-3,
             self.n_moves * 1e-3,
             self.acc_rate,
             clock() - self.start)

    done = property(fget=getDone, doc="is done")
    acc_rate = property(fget=getAccRate, doc="acceptance rate")
    progress = property(fget=getProgress, doc="progress")
    length = property(fget=getLength, doc="length")



class Kernel(rv_discrete):
    '''
       Wrapper class for Markov kernel.
    '''
    def __init__(self, f, name='Markov kernel', longname='Markov kernel.'):
        '''
            Constructor.
            @param f log probability mass function of the invariant distribution 
            @param name name
            @param longname longname
        '''
        rv_discrete.__init__(self, name=name, longname=longname)

        ## log probability mass function of the invariant distribution 
        self.f = f
        ## dimension
        self.d = f.d

    @classmethod
    def setup(cls, f, q):
        return cls(f, q)

    def rvs(self, x, log_f_x=None):
        '''
            Draw from kernel k(x,\cdot)
        '''
        if log_f_x is None: log_f_x = self.f.lpmf(x)
        return self._rvs(x, log_f_x)

    def adapt(self, mean, cov):
        return

    def proposal(self, x, Index):
        Y = x.copy()
        for index in Index:
            Y[index] = Y[index] ^ True
        log_f_Y = self.f.lpmf(Y)
        return Y, log_f_Y
    
    def getPermutation(self):
        perm = range(self.d)
        for i in reversed(range(1, self.d)):
            # pick an element in p[:i+1] with which to exchange p[i]
            j = random.randint(0, i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm


class Gibbs(Kernel):
        '''
            Gibbs kernel
        '''

        def __init__(self, f, q=1.0, name='Gibbs kernel', longname='Gibbs kernel.'):
            '''
                Constructor.
                @param f log probability mass function of the invariant distribution 
            '''
            Kernel.__init__(self, f, name, longname)
            self.q = q

        def _rvs(self, x, log_f_x):
            '''
                Draw from Gibbs kernel k(x,\cdot)
            '''
            Y, log_f_Y = self.proposal(x, Index=[random.randint(1, self.d)])

            if random.random() < 1.0 / (1.0 + exp(log_f_x - log_f_Y)):
                return Y, log_f_Y, True, True
            else:
                return x, log_f_x, False, True


class SymmetricMetropolisHastings(Gibbs):
        '''
            Symmetric Metropolis-Hastings kernel
        '''

        def __init__(self, f, q, name='Symmetric Metropolis-Hastings kernel', longname='Symmetric Metropolis-Hastings kernel.'):
            '''
                Constructor.
                @param f log probability mass function of the invariant distribution 
            '''
            Gibbs.__init__(self, f, q, name=name, longname=longname)

        def _rvs(self, x, log_f_x):
            '''
                Draw from Symmetric Metropolis-Hastings kernel k(x,\cdot)
            '''
            Y, log_f_Y = self.proposal(x, Index=self.getRandomSubset())
            if random.random() < exp(log_f_Y - log_f_x):
                return Y, log_f_Y, True, True
            else:
                return x, log_f_x, False, True

        def getRandomSubset(self):
            k = min(geom.rvs(1.0/self.q), self.d)
            if k < 5:
                Index = []
                while len(Index) < k:
                    n = random.randint(1, self.d)
                    if not n in Index: Index.append(n)
            else:
                Index = self.getPermutation()[:k]
            return Index


class AdaptiveMetropolisHastings(Kernel):
        '''
            Symmetric Metropolis-Hastings kernel
        '''

        def __init__(self, f, q=1.0, name='Adaptive Metropolis-Hastings kernel', longname='Adaptive Metropolis-Hastings kernel.'):
            '''
                Constructor.
                @param f log probability mass function of the invariant distribution 
            '''
            Kernel.__init__(self, f=f, name=name, longname=longname)
            self.mean = 0.5 * ones(self.d)
            self.W = eye(self.d)
            self.k = 0
            self.perm = range(self.d)
            self.delta = 0.01
            self.xlambda = 0.01

        def adapt(self, mean, cov):
            self.mean = mean
            self.W = linalg.inv(cov + self.xlambda * eye(self.d))

        def _rvs(self, x, log_f_x):
            '''
                Draw from Metropolised Gibbs kernel k(x,\cdot)
            '''
            if self.k == self.d:
                self.perm = self.getPermutation()
                self.k = 0
            j = self.perm[self.k]
            self.k += 1

            not_j = [i for i in self.perm if not i == j]
            v = dot(self.W[j, not_j], x[not_j] - self.mean[not_j])
            psi = self.mean[j] - v / self.W[j, j]

            q = max(min(psi, 1 - self.delta), self.delta)

            # return if there is no mutation
            if (random.random() < q) == x[j]: return x, log_f_x, False, False

            Y, log_f_Y = self.proposal(x, Index=[j])

            if Y[j]:
                r = (1 - q) / q
            else:
                r = q / (1 - q)

            if random.random() < exp(log_f_Y - log_f_x) * r:
                return Y, log_f_Y, True, True
            else:
                return x, log_f_x, False, True
