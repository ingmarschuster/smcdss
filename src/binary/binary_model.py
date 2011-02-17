#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

import scipy.stats
from numpy import *
import utils

class Binary(scipy.stats.rv_discrete):
    '''
        A multivariate Bernoulli.
    '''
    def __init__(self, name='binary', longname='A multivariate Bernoulli.'):
        '''
            Constructor.
            @param name name
            @param longname longname
        '''
        scipy.stats.rv_discrete.__init__(self, name=name, longname=longname)
        self.f_lpmf = None
        self.f_rvs = None
        self.f_rvslpmf = None
        self.param = None

    def pmf(self, gamma, job_server=None):
        ''' Probability mass function.
            @param gamma binary vector
        '''
        return exp(self.lpmf(gamma, job_server=job_server))

    def lpmf(self, gamma, job_server=None):
        ''' Log-probability mass function.
            @param gamma binary vector
            @param job_server parallel python job server
        '''
        if len(gamma.shape) == 1: gamma = gamma[newaxis, :]
        size = gamma.shape[0]
        ncpus, job_server = _check_job_server(size, job_server)
        L = empty(size, dtype=float)

        if not job_server is None:
            # start jobs
            print 'run %i jobs' % ncpus
            jobs = _parts_job_server(size, ncpus)
            for i, (start, end) in enumerate(jobs):
                jobs[i].append(
                    job_server.submit(
                    func=self.f_lpmf,
                    args=(gamma[start:end], self.param),
                    modules=('numpy', 'scipy.linalg', 'utils')))

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                L[start:end] = job()
        else:
            # no job server
            L = self.f_lpmf(gamma, param=self.param)

        if size == 1: return L[0]
        else: return L

    def rvs(self, size=1, job_server=None):
        ''' Sample random variables.
            @param size number of variables
            @return random variable
        '''
        ncpus, job_server = _check_job_server(size, job_server)
        Y = empty((size, self.d), dtype=bool)
        U = random.random((size, self.d))

        if not job_server is None:
            # start jobs
            jobs = _parts_job_server(size, ncpus)
            for i, (start, end) in enumerate(jobs):
                jobs[i].append(
                    job_server.submit(
                    func=self.f_rvs,
                    args=(U[start:end], self.param),
                    modules=('numpy', 'scipy.linalg', 'utils')
                    ))

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                Y[start:end] = job()
        else:
            # no job server
            Y = self.f_rvs(U=U, param=self.param)

        if size == 1: return Y[0]
        else: return Y

    def rvslpmf(self, size=1, job_server=None):
        ''' Sample random variables and computes the probabilities.
            @param size number of variables
            @return random variable
        '''
        ncpus, job_server = _check_job_server(size, job_server)
        Y = empty((size, self.d), dtype=bool)
        U = random.random((size, self.d))
        L = empty(size, dtype=float)

        if not job_server is None:
            # start jobs
            jobs = _parts_job_server(size, ncpus)
            for i, (start, end) in enumerate(jobs):
                jobs[i].append(
                    job_server.submit(
                    func=self.f_rvslpmf,
                    args=(U[start:end], self.param),
                    modules=('numpy', 'scipy.linalg', 'utils'),
                    depfuncs=(self.f_rvs, self.f_lpmf)))

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                Y[start:end], L[start:end] = job()
        else:
            # no job server
            Y, L = self.f_rvslpmf(U=U, param=self.param)

        if size == 1: return Y[0], L[0]
        else: return Y, L

    def rvstest(self, n):
        '''
            Prints the empirical mean and correlation to stdout.
            @param n sample size
        '''
        sample = data()
        sample.sample(self, n)
        return format(sample.mean, 'sample (n = %i) mean' % n) + '\n' + \
               format(sample.cor, 'sample (n = %i) correlation' % n)

    def marginals(self):
        '''
            Get string representation of the marginals. 
            @remark Evaluation of the marginals requires exponential time. Do not do it.
            @return a string representation of the marginals 
        '''
        sample = data()
        for dec in range(2 ** self.d):
            bin = dec2bin(dec, self.d)
            sample.append(bin, self.lpmf(bin))
        return sample

def _parts_job_server(size, ncpus):
    return [[i * size // ncpus, min((i + 1) * size // ncpus + 1, size)] for i in range(ncpus)]

def _check_job_server(size, job_server):
    ncpus = 0
    if size == 1: job_server = None
    if not job_server is None:
        ncpus = job_server.get_ncpus()
        if ncpus == 0: job_server = None
    return ncpus, job_server
