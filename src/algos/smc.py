#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-12-02 20:52:23 +0100 (mar., 30 nov. 2010) $
    $Revision: 1 $
'''

from numpy import *
from time import clock
from operator import setitem
from scipy.weave import inline, converters
from sys import stdout

from binary import *
from auxpy.data import data

if system() == 'Linux':    hasWeave = True
else:                      hasWeave = False

CONST_PRECISION = 1e-8


def smc(param, verbose=True):

    print 'running smc...\n'

    ps = ParticleSystem(param)

    # run sequential MC scheme
    while ps.rho < 1.0:

        ps.fit_proposal()
        ps.resample()
        ps.move()
        ps.next_step()

    print "\ndone in %.3f seconds.\n" % (clock() - ps.start)

    return str(ps)


class ParticleSystem(object):

    def __init__(self, param, verbose=True):
        '''
            Constructor.
            @param f target function
            @param verbose verbose
        '''
        self.verbose = verbose
        self.start = clock()

        ## target function
        self.f = param['f']
        ## proposal model
        self.prop = HybridBinary.uniform(self.f.d)

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
        self.log_prop = ones(self.n, dtype=float)
        ## array of ids
        self.id = [0] * self.n

        ## annealing parameter
        self.rho = 0
        ## tau
        self.tau = param['smc_tau']

        ## move step counter
        self.n_moves = 0
        ## target function evaluation counter
        self.n_f_evals = 0

        ##
        self.eps = param['smc_eps']
        ##
        self.delta = param['smc_delta']
        ##
        self.xi = param['smc_xi']

        self.__k = array([2 ** i for i in range(self.d)])

        # initialize particle system
        for i in xrange(self.n):
            self.X[i] = self.prop.rvs()
            self.log_f[i] = self.f.lpmf(self.X[i])
            self.id[i] = self.getId(self.X[i])

        # do first step
        self.next_step()

    def getId(self, x):
        '''
            Assigns a unique id to x.
            @param x binary vector.
            @return id
        '''
        return dot(self.__k, array(x, dtype=int))

    def getEss(self, alpha=None):
        '''
            Computes the effective sample size (ess).
            @param alpha advance of the geometric bridge
            @return ess
        '''
        if alpha is None: w = self.log_W
        else:             w = alpha * self.log_f
        w = exp(w - w.max())
        w /= w.sum()
        return 1 / (self.n * pow(w, 2).sum())

    def getParticleDiversity(self):
        '''
            Computes the particle diversity.
            @return particle diversity
        '''
        dic = {}
        map(setitem, (dic,)*self.n, self.id, [])
        return len(dic.keys()) / float(self.n)

    def next_step(self):
        '''
            Computes an advance of the geometric bridge such that ess = tau and updates the log weights.
        '''
        l = 0.0; u = 1.0 - self.rho
        alpha = 0.05

        # run bisectional search
        for iter in range(30):
            if self.getEss(alpha) < self.tau:
                u = alpha; alpha = 0.5 * (alpha + l)
            else:
                l = alpha; alpha = 0.5 * (alpha + u)
            if abs(l - u) < CONST_PRECISION or l > 1.0: break

        # update rho and and log weights
        self.rho += alpha
        self.log_W += alpha * self.log_f

        if self.verbose: print 'progress %.1f' % (100 * self.rho) + '%'
        print '\n' + str(self) + '\n'

    def fit_proposal(self):
        '''
            Adjust the proposal model to the particle system.
        '''
        sample = data(self.X, self.log_W)
        sample.distinct()
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

    def __str__(self):
        sample = data(self.X, self.log_W)
        mean = '[' + ', '.join(['%.4f' % x for x in sample.getMean(weight=True)]) + ']'
        return '%s;%.3f;%.3f' % (mean, self.n_f_evals / 1000.0, clock() - self.start)

    def move(self):
        '''
            Moves the particle system according to an independent Metropolis-Hastings kernel
            to fight depletion of the particle system.
        '''

        prev_pD = 0
        for iter in range(10):

            self.n_moves += 1
            if self.verbose:
                bars = 20
                drawn = 0
                print 'move.'
                stdout.write('[' + bars * ' ' + "]" + "\r" + "[")

            # pass particle system through transition kernel
            n_acceptance = 0
            for index in range(self.n):
                n_acceptance += self.ind_MH(index)

                if self.verbose:
                    n = bars * (index + 1) / (self.n) - drawn
                    if n > 0:
                        stdout.write(n * "-")
                        stdout.flush()
                        drawn += n

            pD = self.pD
            if self.verbose: print "\naR: %.3f, pD: %.3f" % (n_acceptance / float(self.n), pD)

            if pD - prev_pD < 0.05 or pD > 0.92: break
            else: prev_pD = pD

    def ind_MH(self, index):
        '''
            Metropolis Hasting kernel with independent proposal.
            
            @param index index of the particle that is argument for the kernel
        '''

        self.n_f_evals += 1

        # generate proposal
        proposal, new_log_prop = self.prop.rvslpmf()
        new_log_f = self.f.lpmf(proposal)
        new_log_pi = self.rho * new_log_f

        # gather current values
        cur_log_pi = self.rho * self.log_f[index]
        cur_log_prop = self.log_prop[index]

        # compute acceptance probability and do MH step
        if rand() < exp(new_log_pi - cur_log_pi + cur_log_prop - new_log_prop):
            self.X[index] = proposal
            self.id[index] = self.getId(proposal)
            self.log_f[index] = new_log_f
            self.log_prop[index] = new_log_prop
            return 1
        else:
            return 0

    def resample(self):
        '''
            Resamples the particle system.
        '''
        if self.verbose: print "resample. ",
        if hasWeave: indices = resample_weave(self.nW)
        else: indices = resample_python(self.nW)

        # move objects according to resampled order
        self.id = [self.id[i] for i in indices]
        for obj in [self.X, self.log_f, self.log_prop]:
            obj = obj[indices]

        if self.verbose: print 'pD: %.3f' % self.pD

        # update log proposal values
        self.log_prop[0] = self.prop.lpmf(self.X[0])
        for i in range(1, self.n):
            if (self.log_prop[i] == self.log_prop[i - 1]).all():
                self.log_prop[i] = self.log_prop[i - 1]
            else:
                self.log_prop[i] = self.prop.lpmf(self.X[i])

        # set weights to unity
        self.log_W = zeros(self.n)

    nW = property(fget=getNWeight, doc="normalized weights")
    pD = property(fget=getParticleDiversity, doc="particle diversity")

def resample_python(w):
    '''
        Computes the particle indices by residual resampling.
        @param w array of weights
    '''
    n = w.shape[0]
    u = random.uniform(size=1, low=0, high=1)
    cnw = n * cumsum(w)
    j = 0
    indices = empty(n, dtype="int")
    for k in xrange(n):
        while cnw[j] < u:
            j = j + 1
        indices[k] = j
        u = u + 1.
    return indices

def resample_weave(w):
    '''
        Computes the particle indices by residual resampling using scypy.weave.
        @param w array of weights
    '''
    code = \
    """
    int j = 0;
    double cumsum = weights(0);
    
    for(int k = 0; k < n; k++)
    {
        while(cumsum < u)
        {
        j++;
        cumsum += weights(j);
        }
        indices(k) = j;
        u = u + 1.;
    }
    """
    n = w.shape[0]
    u = float(random.uniform(size=1, low=0, high=1)[0])
    weights = n * w

    indices = zeros(n, dtype="int")
    inline(code, ['u', 'n', 'weights', 'indices'], \
                 type_converters=converters.blitz, compiler='gcc')
    return indices
