#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Sequential Monte Carlo on binary spaces.
"""
import scipy
import subprocess

"""
@namespace ibs.smc
$Author$
$Rev$
$Date$
@details
"""

import time, datetime, sys, operator
import numpy

import ibs
import utils

class smc():
    """ Auxiliary class. """
    header = ['NO_EVALS', 'TIME']
    @staticmethod
    def run(v):
        return integrate_smc(v)

def integrate_smc(param):

    sys.stdout.write('running smc')

    ps = ParticleSystem(param)

    # run sequential MC scheme
    while ps.rho < 1.0:

        ps.fit_proposal()
        ps.augment()
        ps.resample()
        ps.move()
        ps.reweight()

    sys.stdout.write('\rsmc completed in %s.\n' % (str(datetime.timedelta(seconds=time.time() - ps.start))))

    return ps.getCsv()


class ParticleSystem(object):

    def __init__(self, v):
        """
            Constructor.
            @param param parameters
            @param verbose verbose
        """
        self.verbose = v['RUN_VERBOSE']
        if self.verbose: sys.stdout.write('...\n\n')

        self.start = time.time()

        if 'cython' in utils.opts: self._resample = utils.cython.resample
        else: self._resample = utils.python.resample

        ## target function
        self.f = v['f']
        self.job_server = v['JOB_SERVER']

        ## proposal model
        self.prop = v['SMC_BINARY_MODEL'].uniform(self.f.d)

        ## dimension of target function
        self.d = self.f.d
        ## number of particles
        self.n = v['SMC_N_PARTICLES']

        ## array of log weights
        self.log_W = numpy.zeros(self.n, dtype=float)
        ## array of log evaluation of the proposal model
        self.log_prop = numpy.empty(self.n, dtype=float)
        ## array of ids
        self.id = [0] * self.n

        ## annealing parameter
        self.rho = 0
        ## move step counter
        self.n_moves = 0
        ## target function evaluation counter
        self.n_f_evals = 0

        ## acceptance rates
        self.r_ac = []
        ## particle diversities
        self.r_pd = []

        ## min mean distance from the boundaries of [0,1] to be considered part of a logistic model
        self.eps = v['SMC_EPS']
        ## min correlation to be considered part of a logistic model
        self.delta = v['SMC_DELTA']

        self.__k = numpy.array([2 ** i for i in xrange(self.d)])

        if self.verbose:
            sys.stdout.write('initializing...')
            t = time.time()
        self.X = self.prop.rvs(self.n, self.job_server)
        self.log_f = self.f.lpmf(self.X, self.job_server)
        self.w = self.getNWeight()
        for i in xrange(self.n): self.id[i] = self.getId(self.X[i])
        if self.verbose: print '\rinitialized in %.2f sec' % (time.time() - t)

        # do first step
        self.reweight()

    def __str__(self):
        return '[' + ', '.join(['%.3f' % x for x in self.getMean()]) + ']'

    def getCsv(self):
        return (','.join(['%.8f' % x for x in self.getMean()]),
                ','.join(['%.3f' % (self.n_f_evals / 1000.0), '%.3f' % (time.time() - self.start)]),
                ','.join(['%.5f' % x for x in self.r_pd]),
                ','.join(['%.5f' % x for x in self.r_ac]),
                ','.join(['%.5f' % x for x in self.log_f]))

    def getMean(self):
        return numpy.dot(self.w, self.X)

    def getId(self, x):
        """
            Assigns a unique id to x.
            @param x binary vector.
            @return id
        """
        return numpy.dot(self.__k, numpy.array(x, dtype=int))

    def getEss(self, alpha=None):
        """ Computes the effective sample size (ess).
            @param alpha advance of the geometric bridge
            @return ess
        """
        if alpha is None: w = self.log_W
        else:             w = alpha * self.log_f
        w = numpy.exp(w - w.max())
        w /= w.sum()
        return 1 / (self.n * pow(w, 2).sum())

    def getParticleDiversity(self):
        """ Computes the particle diversity.
            @return particle diversity
        """
        dic = {}
        map(operator.setitem, (dic,)*self.n, self.id, [])
        return len(dic.keys()) / float(self.n)

    def reweight(self):
        """ Computes an advance of the geometric bridge such that ess = tau and updates the log weights. """
        l = 0.0; u = 1.05 - self.rho
        alpha = min(0.05, u)

        tau = 0.9

        # run bisectional search
        for iter in xrange(30):

            if self.getEss(alpha) < tau:
                u = alpha; alpha = 0.5 * (alpha + l)
            else:
                l = alpha; alpha = 0.5 * (alpha + u)

            if abs(l - u) < ibs.CONST_PRECISION or self.rho + l > 1.0: break

        # update rho and and log weights
        if self.rho + alpha > 1.0: alpha = 1.0 - self.rho
        self.rho += alpha
        self.log_W = alpha * self.log_f

        if self.verbose:
            utils.format.progress(ratio=self.rho, text='\n')
            print '\n' + str(self) + '\n'
        self.w = self.getNWeight()

    def fit_proposal(self):
        """ Adjust the proposal model to the particle system.
            @todo sample.distinct could ba activated for speedup
        """
        if self.verbose:
            sys.stdout.write('fitting proposal...')
            t = time.time()
        sample = utils.data.data(self.X, self.log_W)
        # sample.distinct()
        self.prop.renew_from_data(sample, job_server=self.job_server, eps=self.eps, delta=self.delta, verbose=False)
        if self.verbose: print '\rfitted proposal in %.2f sec' % (time.time() - t)

    def getNWeight(self):
        """
            Get the normalized weights.
            @return normalized weights
        """
        w = numpy.exp(self.log_W - max(self.log_W))
        return w / w.sum()

    def getSystemStructure(self):
        """
            Gather a summary of how many particles are n-fold in the particle system.
        """
        id_set = set(self.id)
        l = [ self.id.count(i) for i in id_set ]
        k = [ l.count(i) * i for i in xrange(1, 101) ]
        return str(k) + ' %i ' % sum(k)

    def getMax(self):
        index = numpy.argmax(self.log_f)
        return self.log_f[index], self.X[index]

    def move(self):
        """ Moves the particle system according to an independent Metropolis-Hastings kernel
            to fight depletion of the particle system.
        """

        prev_pD = 0
        self.r_ac += [0]
        for iter in xrange(10):
            self.n_moves += 1
            accept = self.kernel()
            self.r_ac[-1] += accept
            pD = self.pD
            if self.verbose: print "moved with aR: %.3f, pD: %.3f" % (accept / float(self.n), pD)
            if pD - prev_pD < 0.04 or pD > 0.93: break
            else: prev_pD = pD

        self.r_ac[-1] /= ((iter + 1) * float(self.n))
        self.r_pd += [pD]

    def augment(self):
        """
            Propagates the particle system via an independent Metropolis Hasting kernel.
            @todo do accept/reject step vectorized
        """

        self.n_f_evals += self.n

        w = self.w()

        # sample
        if self.verbose:
            sys.stdout.write('sampling...')
            t = time.time()
        Y, log_prop_Y = self.prop.rvslpmf(self.n, self.job_server)
        if self.verbose: print '\rsampled in %.2f sec' % (time.time() - t)

        # evaluate
        if self.verbose:
            sys.stdout.write('evaluating...')
            t = time.time()
        log_f_Y = self.f.lpmf(Y, self.job_server)
        if self.verbose: print '\revaluated in %.2f sec' % (time.time() - t)

        # move
        log_pi_Y = self.rho * log_f_Y
        log_pi_X = self.rho * self.log_f
        log_prop_X = self.log_prop

        p = numpy.minimum(numpy.exp(log_pi_Y - log_pi_X + log_prop_X - log_prop_Y), numpy.ones(self.n))
        self.w = numpy.concatenate((w * (numpy.ones(self.n) - p), w * p))
        self.X = numpy.concatenate((self.X, Y))
        self.m = 2 * self.n

    def reduce(self):
        pass #select(self.n, self.w, 0, self.m)

    def kernel(self):
        """
            Propagates the particle system via an independent Metropolis Hasting kernel.
            @todo do accept/reject step vectorized
        """

        self.n_f_evals += self.n

        # sample
        if self.verbose:
            sys.stdout.write('sampling...')
            t = time.time()
        Y, log_prop_Y = self.prop.rvslpmf(self.n, self.job_server)
        if self.verbose: print '\rsampled in %.2f sec' % (time.time() - t)

        # evaluate
        if self.verbose:
            sys.stdout.write('evaluating...')
            t = time.time()
        log_f_Y = self.f.lpmf(Y, self.job_server)
        if self.verbose: print '\revaluated in %.2f sec' % (time.time() - t)

        # move
        log_pi_Y = self.rho * log_f_Y
        log_pi_X = self.rho * self.log_f
        log_prop_X = self.log_prop

        accept = numpy.random.random(self.n) < numpy.exp(log_pi_Y - log_pi_X + log_prop_X - log_prop_Y)
        self.X[accept] = Y[accept]
        self.log_f[accept] = log_f_Y[accept]
        self.log_prop[accept] = log_prop_Y[accept]
        for index in xrange(self.n):
            if accept[index]:
                self.id[index] = self.getId(Y[index])
        return accept.sum()

    def resample(self):
        """ Resamples the particle system. """

        if self.verbose:
            t = time.time()
            sys.stdout.write('resampling...')
        self.w = self.getNWeight()
        indices = self._resample(self.w, numpy.random.random())

        # move objects according to resampled order
        self.id = [self.id[i] for i in indices]
        self.X = self.X[indices]
        self.log_f = self.log_f[indices]

        pD = self.pD

        # update log proposal values
        if not self.job_server is None and self.job_server.get_ncpus() > 1:
            self.log_prop = self.prop.lpmf(self.X, self.job_server)
        else:
            self.log_prop[0] = self.prop.lpmf(self.X[0])
            for i in xrange(1, self.n):
                if (self.log_prop[i] == self.log_prop[i - 1]).all():
                    self.log_prop[i] = self.log_prop[i - 1]
                else:
                    self.log_prop[i] = self.prop.lpmf(self.X[i])

        if self.verbose:
            print '\rresampled in %.2f sec, pD: %.3f' % (time.time() - t, pD)

    pD = property(fget=getParticleDiversity, doc="particle diversity")




def select_recursive(w, n, l=None, u=None):
    """ Selects kappa via recursive search. 
        @param w weights
        @param n target sum
        @param l lower bound
        @param u upper bound
        @return kappa s.t. sum_j^m min(w_j / kappa, 1) <= n
    """
    if u is None:
        w.sort()
        u = w.shape[0] - 1
        l = 0
    if l == u: return w[l]
    q = int(l + 0.5 * (u - l))
    if numpy.minimum(w / w[q], numpy.ones(w.shape[0])).sum() > n:
        return select_recursive(w, n, q + 1, u)
    else:
        return select_recursive(w, n, l, q)

def select_iterative(w, n):
    """ Selects kappa via bisectional search. 
        @param w weights
        @param n target sum
        @return kappa s.t. sum_j^m min(w_j / kappa, 1) <= n
    """
    w.sort()
    m = w.shape[0]
    l, u = 0, m - 1
    while True:
        q = int(l + 0.5 * (u - l))
        if numpy.minimum(w / w[q], numpy.ones(w.shape[0])).sum() > n:
            l = q + 1
        else:
            u = q
        if u == l: return w[l]

def select_linear(w, n):
    """ Selects kappa via backward linear search.
        @param w weights
        @param n target sum
        @return kappa s.t. sum_j^m min(w_j / kappa, 1) <= n
    """
    w.sort()
    m = w.shape[0]
    for i in xrange(m - 1, -1, -1):
        if numpy.minimum(w / w[i], numpy.ones(m)).sum() > n: return w[min(i + 1, m - 1)]

def resample_strat(w, n, f_select=select_iterative):
    """ Resamples a vector of weights.
        @param w weights
        @param n size of resampled vector
        @param f_select selection algorithm
        @return w resampled weights
        @return index resampled indices
    """
    # select smallest value kappa s.t. sum_j^m min(w_j / kappa, 1) <= n
    kappa = f_select(w.copy(), n)

    # nothing to do
    if kappa is None: return w, range(w.shape[0])

    # compute theshold value c s.t. sum_j^m min(c * w_j, 1) = n
    A = (w >= kappa).sum()
    B = (w[numpy.where(w < kappa)[0]]).sum()
    c = (n - A) / B

    # indices of weights to be copied
    index_copy = numpy.where(w * c >= 1)[0]

    # indices of weights to be resampled from
    index_resample = numpy.where(w * c < 1)[0]

    # number of particles to be resampled
    l = n - index_copy.shape[0]

    # weight to assigned to every index on average
    k = w[index_resample].sum() / l

    # random seed
    u = numpy.random.random()*k

    index, j = numpy.zeros(l, dtype=int), 0
    for i in index_resample:
        u -= w[i]
        if u < 0:
            index[j] = i
            j += 1
            u += k

    w = numpy.concatenate((w[index_copy], numpy.ones(l) / c))
    index = numpy.concatenate((index_copy, index))
    return w, index

def get_importance_weights(m=5000, mean=5, sd=5):
    """ Samples from a normal with given mean and standard deviation
        as instrumental function for a standard normal.
        @param m size of weighted sample
        @param mean mean of proposal
        @param sd standard deviation of proposal
        @return w weights
        @return x sample
    """
    x = numpy.random.normal(size=m) * sd + mean
    w = numpy.exp(((1.0 - sd * sd) * x * x - 2.0 * mean * x) / (2.0 * sd * sd))
    w /= w.sum()
    return w, x

def test_selection(m=5000, n=2500, mean=5, sd=5):
    """ Tests the selection algorithms.
        @param m size of weighted sample
        @param n size of resampled system
        @param mean mean of proposal
        @param sd standard deviation of proposal
    """
    w, x = get_importance_weights(m, mean, sd)
    for f in [select_linear, select_iterative, select_recursive]:
        t = time.clock()
        v = f(w, n)
        print '%s value: %.8f  time: %.5f' % (f.__name__.ljust(17), v, time.clock() - t)

def test_resample(f=resample_strat, m=2500, n=500, mean=5, sd=5):
    """ Tests the resampling algorithm.
        @param f resampling algorithm
        @param m size of weighted sample
        @param n size of resampled system
        @param mean mean of proposal
        @param sd standard deviation of proposal
    """
    w1, x1 = get_importance_weights(m, mean, sd)
    w2, index = f(w1.copy(), n)
    x2 = x1[index]

    print '\tweighted  resampled'
    print 'mean\t%.5f  %.5f' % (numpy.dot(x1, w1), numpy.dot(x2, w2))

    v = dict(x1=','.join(['%.20f' % k for k in x1]),
             w1=','.join(['%.20f' % k for k in w1]),
             x2=','.join(['%.20f' % k for k in x2]),
             w2=','.join(['%.20f' % k for k in w2]),
             mean='%.20f' % mean,
             sd='%.20f' % sd,
             m='%d' % m,
             n='%d' % n
        )

    f = open('/home/cschafer/Bureau/tmp.R', 'w')
    f.write(
        '''
        x1=c(%(x1)s)
        w1=c(%(w1)s)
        w1=w1/sum(w1)
        
        x2=c(%(x2)s)
        w2=c(%(w2)s)
        w2=w2/sum(w2)
        
        pdf('/home/cschafer/Bureau/tmp.pdf', width=12, height=4)
        par(mfrow=c(1,3))
        p=density(x=x1, kernel='rectangular')
        plot(p$x, p$y, type='l', xlab='', ylab='', main='original sample')
        lines(p$x, dnorm(p$x, mean=%(mean)s, sd=%(sd)s), type='l', col='blue')
        abline(v = %(mean)s, col = "red")
         
        p=density(x=x1, weights=w1, kernel='rectangular', adjust=0.1)
        plot(p$x, p$y, type='l', xlim=c(-4,4), xlab='', ylab='', main=paste('weighted sample, m=',%(m)s))
        lines(p$x, dnorm(p$x), type='l', xlim=c(-4,4), col='blue')
        abline(v = 0, col = "red")
        
        p=density(x=x2, weights=w2, kernel='rectangular', adjust=0.5)         
        plot(p$x, p$y, type='l', xlim=c(-4,4), xlab='', ylab='', main=paste('resampled version, n=',%(n)s))
        lines(p$x, dnorm(p$x), type='l', xlim=c(-4,4), col='blue')
        abline(v = 0, col = "red") 
        
        dev.off()
        ''' % v
    )
    f.close()
    subprocess.Popen(['R', 'CMD', 'BATCH', '--vanilla', '/home/cschafer/Bureau/tmp.R']).wait()

def main():
    test_resample(m=5000, n=2500, mean=5, sd=5)

if __name__ == "__main__":
    main()
