#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Binary model.
"""

"""
@namespace binary.binary_model
$Author$
$Rev$
$Date$
@details
"""

from binary import *

class Binary(stats.rv_discrete):
    """ Binary model. """
    
    def __init__(self, name='binary', longname='Binary model.'):
        """
            Constructor.
            @param name name
            @param longname longname
        """
        stats.rv_discrete.__init__(self, name=name, longname=longname)
        self.f_lpmf = None
        self.f_rvs = None
        self.f_rvslpmf = None
        self.param = None
        self._v2m_perm = None
        self._m2v_perm = None

    def pmf(self, gamma, job_server=None):
        """ 
            Probability mass function.
            @param gamma binary vector
        """
        return numpy.exp(self.lpmf(gamma, job_server=job_server))

    def lpmf(self, gamma, job_server=None):
        """ 
            Log-probability mass function.
            @param gamma binary vector
            @param job_server parallel python job server
        """
        if len(gamma.shape) == 1: gamma = gamma[numpy.newaxis, :]
        size = gamma.shape[0]
        if self.v2m_perm is not None:
            gamma = gamma[:, self.v2m_perm]
        ncpus, job_server = _check_job_server(size, job_server)
        L = numpy.empty(size, dtype=float)

        if not job_server is None:
            # start jobs
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
        """ 
            Sample random variables.
            @param size number of variables
            @return random variable
        """
        ncpus, job_server = _check_job_server(size, job_server)
        Y = numpy.empty((size, self.d), dtype=bool)
        U = self.rvsbase(size)

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
            Y = self.f_rvs(U, param=self.param)

        if self.m2v_perm is not None:
            Y = Y[:, self.m2v_perm]

        if size == 1: return Y[0]
        else: return Y

    def rvslpmf(self, size=1, job_server=None):
        """ 
            Sample random variables and computes the probabilities.
            @param size number of variables
            @return random variable
        """
        ncpus, job_server = _check_job_server(size, job_server)
        Y = numpy.empty((size, self.d), dtype=bool)
        U = self.rvsbase(size)
        L = numpy.empty(size, dtype=float)

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
            Y, L = self.f_rvslpmf(U, param=self.param)

        if self.m2v_perm is not None:
            Y = Y[:, self.m2v_perm]

        if size == 1: return Y[0], L[0]
        else: return Y, L

    def rvsbase(self, size):
        return numpy.random.random((size, self.d))

    def rvstest(self, n):
        """
            Prints the empirical mean and correlation to stdout.
            @param n sample size
        """
        sample = utils.data.data()
        sample.sample(self, n)
        return utils.format.format(sample.mean, 'sample (n = %i) mean' % n) + '\n' + \
               utils.format.format(sample.cor, 'sample (n = %i) correlation' % n)

    def marginals(self):
        """
            Get string representation of the marginals. 
            @remark Evaluation of the marginals requires exponential time. Do not do it.
            @return a string representation of the marginals 
        """
        sample = utils.data.data()
        for dec in range(2 ** self.d):
            bin = utils.format.dec2bin(dec, self.d)
            sample.append(bin, self.lpmf(bin))
        return sample

    def _getD(self):
        return self.getD()

    def getD(self):
        """ Get dimension.
            @return dimension 
        """
        return 0

    def _getRandom(self):
        return self.getRandom()

    def getRandom(self):
        """ Get index list of random components.
            @return index list 
        """
        return []

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
    d = property(fget=_getD, doc="dimension")
    r = property(fget=_getRandom, doc="random components")

def _parts_job_server(size, ncpus):
    """
        Partitions a load to pass it to multiple cpus.
        @param size sample size
        @param ncpus number of cpus
        @return partition of 0,...,size
    """
    return [[i * size // ncpus, min((i + 1) * size // ncpus + 1, size)] for i in range(ncpus)]

def _check_job_server(size, job_server):
    """
        Checks and modifies the job_server.
        @param size sample size
        @param job_server
        @return ncpus number of cpus
        @return job_server modified job_server
    """
    ncpus = 0
    if size == 1: job_server = None
    if not job_server is None:
        ncpus = job_server.get_ncpus()
        if ncpus == 0: job_server = None
    return ncpus, job_server
