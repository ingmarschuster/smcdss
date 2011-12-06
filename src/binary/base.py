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
                    func=self.pp_depfuncs['lpmf'],
                    args=(gamma[start:end], self.param),
                    modules=self.pp_modules
                ))

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                L[start:end] = job()
        else:
            # no job server
            L = self.pp_depfuncs['lpmf'](gamma, param=self.param)

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
                    func=self.pp_depfuncs['rvs'],
                    args=(U[start:end], self.param),
                    modules=self.pp_modules
                ))

            # wait and retrieve results
            job_server.wait()
            for start, end, job in jobs:
                Y[start:end] = job()
        else:
            # no job server
            Y = self.pp_depfuncs['rvs'](U, param=self.param)

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

        print self.pp_modules
        print self.pp_depfuncs

        if not job_server is None:
            # start jobs
            jobs = _parts_job_server(size, ncpus)
            for i, (start, end) in enumerate(jobs):
                jobs[i].append(
                    job_server.submit(
                    func=self.wrapper.rvslpmf,
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

    def rvstest(self, n):
        """
            Prints the empirical mean and correlation to stdout.
            \param n sample size
        """
        if self.hasPP:
            job_server = pp.Server(ncpus='autodetect', ppservers=())
            print 'running %d cpus' % job_server.get_ncpus()
        else:
            job_server = None
        X, w = self.rvslpmf(n, job_server)
        #assert (w == self.lpmf(X)).all()
        sample = utils.data.data(X=X, w=w)
        return utils.format.format(sample.mean, 'sample (n = %i) mean' % n) + \
               utils.format.format(sample.cor, 'sample (n = %i) correlation' % n)

    def marginals(self):
        """
            Get string representation of the marginals. 
            \remark Evaluation of the marginals requires exponential time. Do not do it.
            \return a string representation of the marginals 
        """
        sample = utils.data.data()
        for dec_vector in range(2 ** self.d):
            bin_vector = utils.format.dec2bin(dec_vector, self.d)
            sample.append(bin_vector, self.lpmf(bin_vector))
        return sample

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
