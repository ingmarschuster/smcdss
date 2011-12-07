#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Binary model.
"""

"""
\namespace binary.base
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

import sys
import numpy
import utils
from utils.data import calc_cov

try: import pp
except: print "error loading parallel python: ", sys.exc_info()[0]

class BaseBinary(object):
    """ Binary parametric family. """

    PRECISION = 1e-5
    MIN_MARGINAL_PROB = 1e-10

    def __init__(self, py_wrapper=None, name='binary family', long_name=__doc__):
        """
            Constructor.
            \param name name
            \param long_name long_name
        """
        self.name = name
        self.long_name = long_name

        self.py_wrapper = py_wrapper
        self.pp_modules = ('numpy',)
        self.pp_depfuncs = ()

        self.hasPP = 'pp' in sys.modules.keys()
        self.param = {'hasPP':self.hasPP}

        #self._v2m_perm = None
        #self._m2v_perm = None

    def __str__(self):
        return 'base class'

    def pmf(self, gamma, job_server=None):
        """ 
            Probability mass function.
            \param gamma binary vector
        """
        return numpy.exp(self.lpmf(gamma, job_server=job_server))

    def lpmf(self, gamma, job_server=None, jobs_per_cpu=1):
        """ 
            Log-probability mass function.
            \param gamma binary vector
            \param job_server parallel python job server
        """
        if len(gamma.shape) == 1: gamma = gamma[numpy.newaxis, :]
        size = gamma.shape[0]
        #if self.v2m_perm is not None:
        #    gamma = gamma[:, self.v2m_perm]
        ncpus, job_server = _check_job_server(size, job_server)
        L = numpy.empty(size, dtype=float)

        if not job_server is None:
            # start jobs
            jobs = _parts_job_server(size, jobs_per_cpu * ncpus)
            for i, (start, end) in enumerate(jobs):
                jobs[i].append(
                    job_server.submit(
                    func=self.py_wrapper.lpmf,
                    args=(gamma[start:end], self.param),
                    modules=self.pp_modules
                ))

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                L[start:end] = job()
        else:
            # no job server
            L = self.py_wrapper.lpmf(gamma, param=self.param)

        if size == 1: return L[0]
        else: return L

    def rvs(self, size=1, job_server=None):
        """ 
            Sample random variables.
            \param size number of variables
            \return random variable
        """
        ncpus, job_server = _check_job_server(size, job_server)
        Y = numpy.empty((size, self.d), dtype=bool)
        U = self._rvsbase(size)

        if not job_server is None:
            # start jobs
            jobs = _parts_job_server(size, ncpus)
            for i, (start, end) in enumerate(jobs):
                jobs[i].append(
                    job_server.submit(
                    func=self.py_wrapper.rvs,
                    args=(U[start:end], self.param),
                    modules=self.pp_modules
                ))

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                Y[start:end] = job()
        else:
            # no job server
            Y = self.py_wrapper.rvs(U, param=self.param)

        #if self.m2v_perm is not None:
        #    Y = Y[:, self.m2v_perm]

        if size == 1: return Y[0]
        else: return Y

    def rvslpmf(self, size=1, job_server=None):
        """ 
            Sample random variables and computes the probabilities.
            \param size number of variables
            \return random variable
        """
        ncpus, job_server = _check_job_server(size, job_server)
        Y = numpy.empty((size, self.d), dtype=bool)
        U = self._rvsbase(size)
        L = numpy.empty(size, dtype=float)

        if not job_server is None:
            # start jobs
            jobs = _parts_job_server(size, ncpus)
            for i, (start, end) in enumerate(jobs):
                jobs[i].append(
                    job_server.submit(
                    func=self.py_wrapper.rvslpmf,
                    args=(U[start:end], self.param),
                    modules=self.pp_modules
                    ))
                # depfuncs=tuple(self.pp_depfuncs.values())

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                Y[start:end], L[start:end] = job()
        else:
            # no job server
            Y, L = self.pp_depfuncs['rvslpmf'](U, param=self.param)

        #if self.m2v_perm is not None:
        #    Y = Y[:, self.m2v_perm]

        if size == 1: return Y[0], L[0]
        else: return Y, L

    def _rvsbase(self, size):
        """ Generates a matrix of random base variables. \param size size """
        return numpy.random.random((size, self.d))

    def rvstest(self, n, start_jobserver='autodetect'):
        """
            Prints the empirical mean and correlation to stdout.
            \param n sample size
        """
        if self.hasPP and start_jobserver:
            job_server = pp.Server(ncpus=start_jobserver, ppservers=())
            print 'rvstest running on %d cpus...\n' % job_server.get_ncpus()
        else:
            job_server = None

        X = self.rvs(n, job_server)

        info = {'mean':repr(numpy.average(X, axis=0)),
                'corr':repr(numpy.corrcoef(X, rowvar=0)), 'n':n}

        return ("sample (n = %(n)d) mean:\n%(mean)s\n" +
                "sample (n = %(n)d) correlation:\n%(corr)s") % info


    def marginals(self, start_jobserver='autodetect'):
        """
            Get string representation of the marginals. 
            \remark Evaluation of the marginals requires exponential time. Do not do it.
            \return a string representation of the marginals 
        """

        # enumerate state space
        X = list()
        for dec_vector in range(2 ** self.d):
            bin_vector = dec2bin(dec_vector, self.d)
            X.append(bin_vector)
        X = numpy.array(X)

        if self.hasPP: job_server = pp.Server(ncpus=start_jobserver, ppservers=())
        else: job_server = None

        lpmf = self.lpmf(X, job_server=job_server)

        # compute weighted mean and correlation
        weights = numpy.exp(lpmf - lpmf.max()); weights /= weights.sum()
        mean = numpy.average(X, axis=0, weights=weights)

        cov = (numpy.dot(X.T, weights[:, numpy.newaxis] * X) - numpy.outer(mean, mean)) / (1 - numpy.inner(weights, weights).sum())
        var = cov.diagonal()
        cor = cov / numpy.sqrt(numpy.outer(var, var))

        info = {'mean': repr(mean), 'corr': repr(cor)}

        return  ("exact mean:\n%(mean)s\n" +
                 "exact correlation:\n%(corr)s") % info

    """
    def getv2m(self):
        return self._v2m_perm

    def setv2m(self, perm):
        self._v2m_perm = numpy.array(perm)
        self._m2v_perm = numpy.argsort(numpy.array(perm))

    def getm2v(self):
        return self._m2v_perm

    def setm2v(self, perm):
        self._m2v_perm = numpy.array(perm)
        self._v2m_perm = numpy.argsort(numpy.array(perm))
        
    v2m_perm = property(fget=getv2m, fset=setv2m, doc="vector to model permutation")
    m2v_perm = property(fget=getm2v, fset=setm2v, doc="model to vector permutation")
    """

    def getD(self):
        """ Get dimension. \return dimension """
        return self._getD()

    def getMean(self):
        """ Get expected value. \return p-vector """
        return self._getMean()

    def getRandom(self):
        """ Get index list of random components. \return index list """
        return self._getRandom()

    d = property(fget=getD, doc="dimension")
    mean = property(fget=getMean, doc="mathematical mean")
    r = property(fget=getRandom, doc="random components")

def _parts_job_server(size, ncpus):
    """
        Partitions a load to pass it to multiple cpus.
        \param size sample size
        \param ncpus number of cpus
        \return partition of 0,...,size
    """
    return [[i * size // ncpus, min((i + 1) * size // ncpus + 1, size)] for i in range(ncpus)]

def _check_job_server(size, job_server):
    """
        Checks and modifies the job_server.
        \param size sample size
        \param job_server
        \return ncpus number of cpus
        \return job_server modified job_server
    """
    ncpus = 0
    if size == 1: job_server = None
    if not job_server is None:
        ncpus = job_server.get_ncpus()
        if ncpus == 0: job_server = None
    return ncpus, job_server

def bin2str(bin):
    """
        Converts a boolean array to a string representation.
        \param bin boolean array 
    """
    return ''.join([str(i) for i in numpy.array(bin, dtype=int)])

def bin2dec(bin):
    """
        Converts a boolean array into an integer.
        \param bin boolean array 
    """
    return long(bin2str(bin), 2)

def dec2bin(n, d=0):
    """
        Converts an integer into a boolean array containing its binary representation.
        \param n integer
        \param d dimension of boolean vector
    """
    bin_vector = []
    while n > 0:
        if n % 2: bin_vector.append(True)
        else: bin_vector.append(False)
        n = n >> 1
    while len(bin_vector) < d: bin_vector.append(False)
    bin_vector.reverse()
    return numpy.array(bin_vector)
