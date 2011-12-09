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

try: import pp
except: print "error loading parallel python: ", sys.exc_info()[0]

class BaseBinary(object):
    """ Binary parametric family. """

    ## Error allowed for Newton iterates.
    PRECISION = 1e-5
    ## Maximum number of Newton iterations.
    MAX_ITERATIONS = 50
    ## Entry gamma_i is considered constant if p_i<MIN_MARGINAL_PROB or 1-p_i<MIN_MARGINAL_PROB
    MIN_MARGINAL_PROB = 1e-10

    def __init__(self, d, name='binary family', long_name=__doc__):
        """
            Constructor.
            \param name name
            \param long_name long_name
        """

        self.name = name
        self.long_name = long_name
        self.d = d

        self.pp_modules = ('numpy',)
        self.pp_depfuncs = ()

        #self._v2m_perm = None
        #self._m2v_perm = None

    def __str__(self):
        return 'base class'

    def pmf(self, Y, job_server=None):
        """ 
            Probability mass function.
            \param Y binary vector
        """
        return self._pmf(Y, job_server=job_server)

    def _pmf(self, Y, job_server=None):
        """ 
            Probability mass function.
            \param Y binary vector
        """
        return numpy.exp(self.lpmf(Y, job_server=job_server))

    def lpmf(self, gamma, job_server=None):
        """ 
            Log-probability mass function.
            \param gamma binary vector
            \param job_server parallel python job server
        """
        if len(gamma.shape) == 1: gamma = gamma[numpy.newaxis, :]
        size = gamma.shape[0]
        #if self.v2m_perm is not None:
        #    gamma = gamma[:, self.v2m_perm]
        ncpus, job_server = BaseBinary._check_job_server(size, job_server)
        L = numpy.empty(size, dtype=float)

        if not job_server is None:
            # start jobs
            jobs = BaseBinary._parts_job_server(size, ncpus)
            for i, (start, end) in enumerate(jobs):
                jobs[i].append(
                    job_server.submit(
                    func=self.py_wrapper.lpmf,
                    args=(gamma[start:end], self),
                    modules=self.pp_modules
                ))

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                L[start:end] = job()
        else:
            # no job server
            L = self.py_wrapper.lpmf(gamma, param=self)

        if size == 1: return L[0]
        else: return L

    def rvs(self, size=1, job_server=None):
        """ 
            Sample random variables.
            \param size number of variables
            \return random variable
        """
        ncpus, job_server = BaseBinary._check_job_server(size, job_server)
        Y = numpy.empty((size, self.d), dtype=bool)
        U = self._rvsbase(size)

        if not job_server is None:
            # start jobs
            jobs = BaseBinary._parts_job_server(size, ncpus)
            for i, (start, end) in enumerate(jobs):
                jobs[i].append(
                    job_server.submit(
                    func=self.py_wrapper.rvs,
                    args=(U[start:end], self),
                    modules=self.pp_modules
                ))

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                Y[start:end] = job()
        else:
            # no job server
            Y = self.py_wrapper.rvs(U, param=self)

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
        ncpus, job_server = BaseBinary._check_job_server(size, job_server)
        Y = numpy.empty((size, self.d), dtype=bool)
        U = self._rvsbase(size)
        L = numpy.empty(size, dtype=float)

        if not job_server is None:
            # start jobs
            jobs = BaseBinary._parts_job_server(size, ncpus)
            for i, (start, end) in enumerate(jobs):
                jobs[i].append(
                    job_server.submit(
                    func=self.py_wrapper.rvslpmf,
                    args=(U[start:end], self),
                    modules=self.pp_modules
                    ))
                # depfuncs=tuple(self.pp_depfuncs.values())

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                Y[start:end], L[start:end] = job()
        else:
            # no job server
            Y, L = self.py_wrapper.self.lpmf(U, param=self.param)

        #if self.m2v_perm is not None:
        #    Y = Y[:, self.m2v_perm]

        if size == 1: return Y[0], L[0]
        else: return Y, L

    def _rvsbase(self, size):
        """ Generates a matrix of random base variables. \param size size """
        return numpy.random.random((size, self.d))

    @classmethod
    def _parts_job_server(cls, size, ncpus):
        """
            Partitions a load to pass it to multiple cpus.
            \param size sample size
            \param ncpus number of cpus
            \return partition of 0,...,size
        """
        return [[i * size // ncpus, min((i + 1) * size // ncpus + 1, size)] for i in range(ncpus)]

    @classmethod
    def _check_job_server(cls, size, job_server):
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


    def rvs_marginals(self, n, ncpus='autodetect'):
        """
            Prints the empirical mean and correlation to stdout.
            \param n sample size
        """
        if 'pp' in sys.modules.keys() and ncpus > 1:
            job_server = pp.Server(ncpus=ncpus, ppservers=())
            print 'rvstest running on %d cpus...\n' % job_server.get_ncpus()
        else: job_server = None

        X = self.rvs(n, job_server)

        return numpy.average(X, axis=0), numpy.corrcoef(X, rowvar=0)

    def exact_marginals(self, ncpus='autodetect'):
        """
            Get string representation of the marginals. 
            \return a string representation of the marginals 
        """

        if 'pp' in sys.modules.keys() and ncpus > 1:
            job_server = pp.Server(ncpus=ncpus, ppservers=())
        else: job_server = None

        X = self.state_space()
        lpmf = self.lpmf(X, job_server=job_server)
        weights = numpy.exp(lpmf - lpmf.max()); weights /= weights.sum()

        return sample2corr(X, weights)

    def state_space(self):
        """ Enumerates the state space. \return array of binary vectors """
        X = list()
        for dec_vector in range(2 ** self.d):
            bin_vector = dec2bin(dec_vector, self.d)
            X.append(bin_vector)
        return numpy.array(X)

    def getMean(self):
        """ Get expected value. \return p-vector """
        return self._getMean()

    def getRandom(self):
        """ Get index list of random components. \return index list """
        return self._getRandom()

    mean = property(fget=getMean, doc="mathematical mean")
    r = property(fget=getRandom, doc="random components")


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

def random_moments(d, eps=0.05, phi=0.5):
    """
        Creates a random cross-moments matrix that is consistent with the
        general constraints on binary data.
        \param d dimension
        \param eps minmum distance to borders of [0,1]
        \param phi parameter in [0,1] where phi=0 means zero correlation
        \return M cross-moment matrix
    """
    M = numpy.diag(eps + (1.0 - 2 * eps) * numpy.random.random(d))
    for i in range(d):
        for j in range(i):
            high = min(M[i, i], M[j, j])
            low = max(M[i, i] + M[j, j] - 1.0, 0)
            M[i, j] = phi * (low + numpy.abs(high - low) * numpy.random.random()) + (1.0 - phi) * M[i, i] * M[j, j]
            M[j, i] = M[i, j]
    return M

def corr2moments(mean, corr):
    """
        Converts a mean vector and correlation matrix to the corresponding
        cross-moment matrix.
        
        \param mena mean vector.
        \param corr correlation matrix
        \return cross-moment matrix
    """
    mean = numpy.minimum(numpy.maximum(mean, 1e-5), 1.0 - 1e-5)
    var = mean * (1.0 - mean)
    M = (corr * numpy.sqrt(numpy.outer(var, var))) + numpy.outer(mean, mean)
    return M

def moments2corr(M):
    """
        Converts a cross-moment matrix to a corresponding pair of mean vector
        and correlation matrix. .
        
        \param M cross-moment matrix
        \return mean vector.
        \return correlation matrix
    """
    mean = M.diagonal()
    adj_mean = numpy.minimum(numpy.maximum(mean, 1e-8), 1.0 - 1e-8)
    var = adj_mean * (1.0 - adj_mean)
    corr = (M - numpy.outer(adj_mean, adj_mean)) / numpy.sqrt(numpy.outer(var, var))
    return mean, corr

def sample2corr(X, weights):

    # compute weighted mean and correlation
    mean = numpy.average(X, axis=0, weights=weights)

    adj_mean = numpy.minimum(numpy.maximum(mean, 1e-8), 1.0 - 1e-8)
    cov = (numpy.dot(X.T, weights[:, numpy.newaxis] * X) - numpy.outer(adj_mean, adj_mean)) / (1 - numpy.inner(weights, weights).sum())
    var = numpy.maximum(cov.diagonal(), 1e-6)
    corr = cov / numpy.sqrt(numpy.outer(var, var))

    return mean, corr


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
