#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-12-02 20:52:23 +0100 (mar., 30 nov. 2010) $
    $Revision: 1 $
'''

from time import clock
from numpy import *
from auxpy.data import data
from sys import stdout

def mcmc(param, verbose=True):
    '''
        Runs a MCMC sampling scheme..
    '''

    print "\nrun markov chain using " + param['mcmc_kernel'] + " kernel."

    start = clock()
    kappa = kernel(param['f'], param['mcmc_kernel'])
    max_time = param['mcmc_max_time']
    max_iter = param['mcmc_max_iter']
    if max_iter < inf: step = int(max_iter / 100.0)
    else: step=10000
    p_sample = data()

    # burn in
    for i in range(step): kappa.rvs()

    if verbose:
        stdout.write("\n" + 101 * " " + "]" + "\r" + "[")
        progress = 0

    # run for maximum time or maximum iterations
    while (clock() - start) / 60.0 < max_time  and kappa.n_iter < max_iter:

        s_sample = data()
        for i in range(step):
            s_sample.append(kappa.rvs())

            # print progress bar
            if verbose:
                if max_iter < inf: progress_next = int(100.0 * kappa.n_iter / float(max_iter))
                else: progress_next = int(10.0 * (clock() - start) / (6.0 * max_time))
                if (clock() - start) / 60.0 >= max_time or kappa.n_iter >= max_iter: progress_next = 100
                stdout.write((progress_next - progress) * "-")
                stdout.flush()
                progress = progress_next

        p_sample.append(s_sample.mean)

    # return results
    mean = '[' + ', '.join(['%.4f' % x for x in p_sample.mean]) + ']'
    return '%s;%.3f;%.3f' % (mean, kappa.n_moves / float(kappa.n_iter), clock() - start)

class kernel():

    def __init__(self, f, kernel_type):

        ##
        self.f = f
        ##
        self.d = f.d
        ##
        self.n_iter = 0
        ##
        self.n_moves = 0
        ##
        self.state = random.random(self.d) > 0.5
        ##
        self.log_state_pi = self.f.lpmf(self.state)

        if kernel_type is 'mh':
            self.kappa = self.kernel_indmh
        elif kernel_type is 'gibbs':
            self.kappa = self.kernel_gibbs
        else:
            raise AttributeError('Kernel "' + kernel_type + '" not recognized. Types should be "mh" or "gibbs"')

    def rvs(self):
        self.kappa()
        self.n_iter += 1
        return self.state

    def kernel_gibbs(self):
        '''
            Gibbs kernel driving the Markov chain
        '''
        proposal = self.state.copy()
        index = random.randint(0, self.d)
        proposal[index] = proposal[index] ^ True
        log_proposal_pi = self.f.lpmf(proposal)
        if random.random() < 1 / (1 + exp(self.log_state_pi - log_proposal_pi)):
            self.log_state_pi = log_proposal_pi
            self.state = proposal
            self.n_moves += 1

    def kernel_indmh(self):
        '''
            MH kernel driving the Markov chain
        '''
        if False:
            components = set()
            v = random.geometric(1 / float(self.scaleneighbors))
            while len(components) < min(v, self.targetDistr.p / 2):
                components.add(random.randint(0, self.targetDistr.p))
        else:
            components = [random.randint(0, self.d)]

        proposal = self.state.copy()
        for index in components:
            proposal[index] = proposal[index] ^ True

        log_proposal_pi = self.f.lpmf(proposal)

        if random.random() < exp(log_proposal_pi - self.log_state_pi):
            self.log_state_pi = log_proposal_pi
            self.state = proposal
            self.n_moves += 1
