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
from scipy.stats import rv_discrete

header = lambda: ['NO_EVALS', 'ACC_RATE' , 'TIME']

def run(param, verbose=True):

    mc = MarkovChain(f=param['f'], kernel=param['mcmc_kernel'], max_calls=param['mcmc_max_calls'])

    mc.burn_in()

    # run for maximum time or maximum iterations
    while not mc.done:
        mc.do_step()
        mc.kernel.adapt(mc.mean, mc.cov)

    return mc.getCsv()


class MarkovChain():

    def __init__(self, f, kernel, max_calls=1e6, step_size=1e5, verbose=True):

        self.start = clock()
        self.verbose = verbose
        self.kernel = kernel.setup(f)
        self.d = f.d
        self.step_size = int(min(step_size, max_calls))
        self.max_calls = float(max_calls)

        self.x = random.random(self.d) > 0.5
        self.log_f_x = self.kernel.f.lpmf(self.x)

        self.n_calls = 0
        self.n_moves = 0
        self.n_steps = 0

        self.mean = zeros(self.d)
        self.cov = zeros((self.d, self.d))

    def burn_in(self, n_burn_in=None):
        if n_burn_in is None: n_burn_in = int(self.max_calls / 100.0)
        if self.verbose:
            print "\n%s: %i steps burn in..." % (self.kernel.name, n_burn_in)
            stdout.write("" + 101 * " " + "]" + "\r" + "[")
            self.n_bars = 0

        for i in range(n_burn_in):
            self.n_calls += 1
            self.x, self.log_f_x = self.kernel.rvs(self.x, self.log_f_x)

            # print progress bar
            if self.verbose:
                if self.n_bars < int(100.0 * self.n_calls / float(n_burn_in)):
                    stdout.write("-")
                    stdout.flush()
                    if self.n_calls >= n_burn_in: break
                    self.n_bars += 1

        if self.verbose:
            print "\n%s: %i steps mcmc..." % (self.kernel.name, self.max_calls)
            stdout.write("" + 101 * " " + "]" + "\r" + "[")
            self.n_bars = 0

    def do_step(self):
        mean = zeros(self.d)
        cov = zeros((self.d, self.d))

        for i in range(self.step_size):
            self.n_calls += 1
            self.x, self.log_f_x = self.kernel.rvs(self.x, self.log_f_x)
            mean += self.x
            cov += dot(self.x[:, newaxis], self.x[newaxis, :])

            # print progress bar
            if self.verbose:
                if self.n_bars < int(100.0 * self.n_calls / self.max_calls):
                    stdout.write("-")
                    stdout.flush()
                    if self.n_calls >= self.max_calls: break
                    self.n_bars += 1

        mean /= float(self.step_size)
        cov /= float(self.step_size)
        r = self.n_steps / float(self.n_steps + 1)
        self.mean = r * self.mean + (1 - r) * mean
        self.cov = r * self.cov + (1 - r) * cov
        self.n_steps += 1

    def getDone(self):
        return self.max_calls <= self.n_calls

    def getCsv(self):
        mean = '\t'.join(['%.8f' % x for x in self.mean])
        return mean , '%.3f\t%.3f\t%.3f' % (self.n_calls / 1000.0, self.n_moves / float(self.n_calls), clock() - self.start)

    done = property(fget=getDone, doc="is done")


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
    def setup(cls, f):
        return cls(f)

    def rvs(self, x, log_f_x=None):
        '''
            Draw from kernel k(x,\cdot)
        '''
        if log_f_x is None: log_f_x = self.f.lpmf(x)
        return self._rvs(x, log_f_x)

    def adapt(self, mean, cov):
        return

    def proposal(self, x, index=None):
        Y = x.copy()
        if index is None: index = random.randint(0, self.d)
        Y[index] = Y[index] ^ True
        log_f_Y = self.f.lpmf(Y)
        return Y, log_f_Y

class Gibbs(Kernel):
        '''
            Gibbs kernel
        '''

        def __init__(self, f, name='Gibbs kernel', longname='Gibbs kernel.'):
            '''
                Constructor.
                @param f log probability mass function of the invariant distribution 
            '''
            Kernel.__init__(self, f, name, longname)

        def _rvs(self, x, log_f_x):
            '''
                Draw from Gibbs kernel k(x,\cdot)
            '''
            Y, log_f_Y = self.proposal(x)

            if random.random() < 1.0 / (1.0 + exp(log_f_x - log_f_Y)):
                return Y, log_f_Y
            else:
                return x, log_f_x


class SymmetricMetropolisHastings(Gibbs):
        '''
            Symmetric Metropolis-Hastings kernel
        '''

        def __init__(self, f):
            '''
                Constructor.
                @param f log probability mass function of the invariant distribution 
            '''
            Gibbs.__init__(self, f, name='Symmetric Metropolis-Hastings kernel', longname='Symmetric Metropolis-Hastings kernel.')


        def _rvs(self, x, log_f_x):
            '''
                Draw from Symmetric Metropolis-Hastings kernel k(x,\cdot)
            '''
            Y, log_f_Y = self.proposal(x)
            if random.random() < exp(log_f_Y - log_f_x):
                return Y, log_f_Y
            else:
                return x, log_f_x

class AdaptiveMetropolisHastings(Kernel):
        '''
            Symmetric Metropolis-Hastings kernel
        '''

        def __init__(self, f):
            '''
                Constructor.
                @param f log probability mass function of the invariant distribution 
            '''
            Kernel.__init__(self, f=f, name='Adaptive Metropolis-Hastings kernel', longname='Adaptive Metropolis-Hastings kernel.')
            self.mean = 0.5 * ones(self.d)
            self.W = eye(self.d)
            self.k = 0
            self.perm = range(self.d)
            self.delta = 0.02
            self.xlambda = 0.001

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
            if (random.random() < q) == x[j]: return x, log_f_x

            Y, log_f_Y = self.proposal(x, index=j)

            if Y[j]:
                r = (1 - q) / q
            else:
                r = q / (1 - q)

            if random.random() < exp(log_f_Y - log_f_x) * r:
                return Y, log_f_Y
            else:
                return x, log_f_x


        def getPermutation(self):
            perm = range(self.d)
            for i in reversed(range(1, self.d)):
                # pick an element in p[:i+1] with which to exchange p[i]
                j = random.randint(0, i + 1)
                perm[i], perm[j] = perm[j], perm[i]
            return perm

