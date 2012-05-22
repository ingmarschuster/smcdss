#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Markov chain
    @namespace algo.mc
    @details Includes the classic Gibbs kernel, Symmetric Metropolis-Hasting kernels
             and Adaptive Metropolis-Hastings kernels.
"""

from binary.base import get_permutation
from utils.auxi import progress
from resample import resample_reductive
import binary.conditionals_logistic
import numpy
import scipy.linalg
import scipy.stats as stats

N_MEMORY = 1000
METROPOLIS_HASTINGS = 1
GIBBS = 2

class MarkovChain(object):
    """ Markov chain. """

    def __init__(self, param, x=None):
        """
            Constructor.
            \param f probability mass function 
            \param kernel Markov kernel
            \param q expected number of bits to be flipped
            \param max_evals maximum number of target evaluations
            \param step_size number of steps stored before updating the estimator
            \param verbose verbose
        """

        ## target function
        self.f = param['f']

        ## dimension of target function
        self.d = self.f.d

        ## verbose outputs
        self.verbose = param['run/verbose']

        ## Markov kernel
        self.adaptive = 0
        q = float(param['mcmc/exp_flips'])
        if param['mcmc/kernel'].lower() == 'symmetric':
            self.kernel = SymmetricMetropolisHastings(f=self.f, q=q)
        if param['mcmc/kernel'].lower() == 'adaptive mh':
            self.kernel = AdaptiveMetropolisHastings(f=self.f, q=q)
        if param['mcmc/kernel'].lower() == 'gibbs':
            self.kernel = Gibbs(self.f)
        if param['mcmc/kernel'].lower() == 'adaptive gibbs':
            self.kernel = AdaptiveGibbs(self.f)

        ## maximum number of target evaluations
        self.max_evals = float(param['mcmc/max_evals'])

        ## number of steps stored before updating the estimator
        if param['mcmc/chunk_size'] in ['', None]:
            self.chunk_size = int(self.max_evals / 100.0)
        else:
            self.chunk_size = int(min(param['mcmc/chunk_size'], self.max_evals))

        ## percentage of run
        self.rho = 0.0

        ## current state
        if x is None:
            self.x = numpy.random.random(self.d) > 0.5
            self.mean = numpy.zeros(self.d)
        else:
            self.x = x
            self.mean = x

        ## log probability of the current state
        self.log_f_x = self.f.lpmf(self.x)

        ## number of moves
        self.n_moves = 0

        ## number of target function evaluations
        self.n_f_evals = 0

        ## number of steps
        self.n_steps = 0

        ## acceptance rates
        self.r_ac = list()

        ## acceptance rates
        self.r_bf = list()

    def __str__(self):
        return '\nmean: %s\nprogress: %.3f %%\nlength: %.3f\nacc_rate: %.3f\nmoves: %.3f\nevals: %.3f' % \
                ('[' + ', '.join(['%.3f' % x for x in self.mean]) + ']',
                 self.rho,
                 self.length * 1e-3,
                 self.acc_rate,
                 self.n_moves * 1e-3,
                 self.n_f_evals * 1e-3)

    def do_step(self, burn_in=False):
        """ Propagate the Markov chain. """
        mean = numpy.zeros(self.d)
        t = 1.0
        moves = 0
        bits = 0

        # init for adaption
        if self.kernel.is_adaptive and not burn_in:
            if self.kernel.type == GIBBS:
                cov = numpy.zeros((self.d, self.d))
            if self.kernel.type == METROPOLIS_HASTINGS:
                weights = list()
                X = list()

        for i in xrange(self.chunk_size):

            x, self.log_f_x, move, evals = self.kernel.rvs(self.x, self.log_f_x)
            self.n_f_evals += evals

            if move:
                mean += t * self.x
                bits += (self.x - x).sum()
                moves += 1
                self.n_moves += 1
                self.x = x

                # stock data for adaption
                if self.kernel.is_adaptive and not burn_in:
                    if self.kernel.type == GIBBS:
                        cov += t * numpy.outer(x, x)
                    if self.kernel.type == METROPOLIS_HASTINGS:
                        weights += [t]
                        X += [x]
                t = 1.0
            else:
                t += 1.0
            if not self.verbose:
                self.rho = progress(ratio=self.n_f_evals / self.max_evals, last_ratio=self.rho)

        # compute update ratio
        r = self.n_steps / float(self.n_steps + 1)

        # update mean
        mean /= float(self.chunk_size)
        self.mean = r * self.mean + (1 - r) * mean

        # acceptance rates
        self.r_ac += [moves / float(self.chunk_size)]
        if not burn_in: self.r_bf += [bits / float(self.chunk_size)]

        # progress
        self.n_steps += 1
        self.rho = self.n_f_evals / self.max_evals

        # adapt kernel
        if self.kernel.is_adaptive and not burn_in:
            if self.kernel.type == METROPOLIS_HASTINGS:
                weights += [self.chunk_size / 20.0]
                X += [0.05 * numpy.ones(self.d) + 0.9 * self.mean]
                if not burn_in: self.kernel.adaptive = 0.5 * self.rho
                self.kernel.adapt(weights, X, r)
            if self.kernel.type == GIBBS:
                cov /= float(self.chunk_size)
                self.kernel.adapt(self.mean, cov, r)

    def get_mean(self):
        return self.mean

    def get_done(self):
        return self.max_evals <= self.n_f_evals

    def get_length(self):
        return self.n_steps * self.chunk_size

    done = property(fget=get_done, doc="is done")
    length = property(fget=get_length, doc="length")


class Kernel(stats.rv_discrete):
    """ Wrapper class for Markov kernels. """

    def __init__(self, f, name='Markov kernel', long_name='Markov kernel.'):
        """
            Constructor.
            \param f log probability mass function of the invariant distribution 
            \param name name
            \param long_name long_name
        """
        super(Kernel, self).__init__(self, name=name)
        self.long_name = long_name
        self.is_adaptive = False

        ## log probability mass function of the invariant distribution 
        self.f = f

        ## dimension
        self.d = f.d

    def rvs(self, x, log_f_x=None):
        """
            Draw from kernel k(x,\cdot)
            \param x current state
            \param log_f_x log probability of current state
        """
        if log_f_x is None: log_f_x = self.f.lpmf(x)
        return self._rvs(x, log_f_x)

    def proposal(self, x, Index):
        Y = x.copy()
        for index in Index:
            Y[index] = Y[index] ^ True
        log_f_Y = self.f.lpmf(Y)
        return Y, log_f_Y


class Gibbs(Kernel):
    """ Gibbs kernel. """

    def __init__(self, f, name='Gibbs kernel', long_name='Gibbs kernel.'):
        """
            Constructor.
            \param f log probability mass function of the invariant distribution 
        """
        super(Gibbs, self).__init__(f, name, long_name)
        self.type = GIBBS

    def _rvs(self, x, log_f_x):
        """
            Draw from Gibbs kernel k(x,\cdot)
            \param x current state
            \param log_f_x log probability of current state
        """
        Y, log_f_Y = self.proposal(x, Index=[numpy.random.randint(low=0, high=self.d)])

        if numpy.random.random() < 1.0 / (1.0 + numpy.exp(log_f_x - log_f_Y)):
            return Y, log_f_Y, True, True
        else:
            return x, log_f_x, False, True


class SwapMetropolisHastings(Kernel):
    """ Swap Metropolis-Hastings kernel. """

    def __init__(self, f, q, name='Swap Metropolis-Hastings kernel', long_name='Swap Metropolis-Hastings kernel.'):
        """
            Constructor.
            \param f log probability mass function of the invariant distribution 
        """
        super(SwapMetropolisHastings, self).__init__(f, name=name, long_name=long_name)
        self.type = METROPOLIS_HASTINGS
        self.q = q

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


class SymmetricMetropolisHastings(Kernel):
    """ Symmetric Metropolis-Hastings kernel. """

    def __init__(self, f, q, name='Symmetric Metropolis-Hastings kernel', long_name='Symmetric Metropolis-Hastings kernel.'):
        """
            Constructor.
            \param f log probability mass function of the invariant distribution 
        """
        super(SymmetricMetropolisHastings, self).__init__(f, name=name, long_name=long_name)
        self.type = METROPOLIS_HASTINGS
        self.q = q

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
            \return subset
        """
        if self.q == 1: return [numpy.random.randint(low=0, high=self.d)]
        k = min(stats.geom.rvs(1.0 / self.q), self.d)
        if k < 5:
            Index = []
            while len(Index) < k:
                n = numpy.random.randint(low=0, high=self.d)
                if not n in Index: Index.append(n)
        else:
            Index = get_permutation(self.d)[:k]
        return Index


class AdaptiveMetropolisHastings(SymmetricMetropolisHastings):

    def __init__(self, f, q, name='Adaptive Metropolis-Hastings kernel',
                 long_name='Adaptive Metropolis-Hastings kernel.'):
        """
            Constructor.
            \param f log probability mass function of the invariant distribution
            \param q expected number of bits to be flipped
            \param name name
            \param long_name long_name
        """
        super(AdaptiveMetropolisHastings, self).__init__(f=f, q=q, name=name, long_name=long_name)
        self.type = METROPOLIS_HASTINGS
        self.is_adaptive = True

        ## proposal distribution
        self.prop = binary.conditionals_logistic.LogisticCondBinary.uniform(self.d)

        ## percentage of adaptive kernel
        self.adaptive = 0.0

        ## weights
        self.weights = numpy.zeros(0)

        ## particles
        self.X = numpy.zeros((0, self.d))

    def adapt(self, weights, X, r):
        if r == 0: return
        weights = numpy.array(weights)
        weights /= weights.sum()
        self.weights = numpy.concatenate((r * self.weights, (1 - r) * weights))
        self.X = numpy.vstack((self.X, numpy.array(X)))
        self.weights /= self.weights.sum()
        self.distinct()

        # update weights and particles
        if self.weights.shape[0] > 2 * N_MEMORY:
            self.weights, index = resample_reductive(w=self.weights, u=numpy.random.random(), n=N_MEMORY)
            self.X = self.X[index]

        self.prop.renew_from_data(self.X, self.weights)

    def distinct(self):
        """ Aggregate weights. """

        weights = self.weights

        # order the data array
        lexorder = numpy.lexsort(numpy.array(self.X).T)

        # check if all entries are equal
        if weights[lexorder[0]] == weights[lexorder[-1]]:
            _X, _weights = numpy.array([self.X[0]]), numpy.array([1.0])
        else:
            _X, _weights = list(), list()

            # loop over ordered data
            x, w = self.X[lexorder[0]], weights[lexorder[0]]

            for i in numpy.append(lexorder[1:], lexorder[0]):
                if (x == self.X[i]).all():
                    w += weights[i]
                else:
                    _X += [x]
                    _weights += [w]
                    x = self.X[i]
                    w = weights[i]

        self.X = numpy.array(_X, dtype=float)
        self.weights = numpy.array(_weights, dtype=float)

    def _rvs(self, x, log_f_x):
        """
            Draw from Symmetric Metropolis-Hastings kernel k(x,\cdot)
        """
        if numpy.random.random() < self.adaptive:
            Y, log_prop_Y = self.prop.rvslpmf()
            log_f_Y = self.f.lpmf(Y)
            log_prop_x = self.prop.lpmf(x)
            prob = numpy.exp(log_f_Y - log_f_x + log_prop_x - log_prop_Y)
        else:
            Y, log_f_Y = self.proposal(x, Index=self.getRandomSubset())
            prob = numpy.exp(log_f_Y - log_f_x)

        if numpy.random.random() < prob:
            return Y, log_f_Y, True, True
        else:
            return x, log_f_x, False, True


class AdaptiveGibbs(Kernel):
    """ Adaptive Gibbs kernel"""

    def __init__(self, f, name='Adaptive Gibbs kernel',
                 long_name='Adaptive Gibbs kernel.'):
        """
            Constructor.
            \param f log probability mass function of the invariant distribution
            \param q expected number of bits to be flipped
            \param name name
            \param long_name long_name
        """
        super(AdaptiveGibbs, self).__init__(f=f, name=name, long_name=long_name)
        self.type = GIBBS
        self.is_adaptive = True
        self.adaptive = False

        self.cov = numpy.zeros((self.d, self.d))

        self.mean = 0.5 * numpy.ones(self.d)
        self.W = numpy.eye(self.d)
        self.k = 0
        self.perm = range(self.d)
        self.delta = 0.01
        self.xlambda = 0.01

    def adapt(self, mean, cov, r):
        """ Adapt the kernel. """
        self.cov = r * self.cov + (1 - r) * cov
        self.mean = mean
        self.W = scipy.linalg.inv(cov + self.xlambda * numpy.eye(self.d))
        self.adaptive = True

    def _rvs(self, x, log_f_x):
        """
            Draw from Metropolised Gibbs kernel k(x,\cdot)
            \param x current state
            \param log_f_x log probability of current state
        """
        if self.k == self.d:
            self.perm = get_permutation(self.d)
            self.k = 0
        j = self.perm[self.k]
        self.k += 1

        if self.adaptive:
            not_j = [i for i in self.perm if not i == j]
            v = numpy.dot(self.W[j, not_j], x[not_j] - self.mean[not_j])
            psi = self.mean[j] - v / self.W[j, j]

            q = max(min(psi, 1 - self.delta), self.delta)

            # return if there is no mutation
            if (numpy.random.random() < q) == x[j]:
                return x, log_f_x, False, False
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
