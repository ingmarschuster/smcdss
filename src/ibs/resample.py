#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Resampling algorithms.
"""

"""
@namespace ibs.resample
$Author: christian.a.schafer@gmail.com $
$Rev: 143 $
$Date: 2011-04-27 19:45:16 +0200 (mer., 27 avr. 2011) $
@details
"""

import subprocess, time
import numpy


#

# Resampling algorithms.

#


def resample_systematic(w, u):
    """
        Computes the particle indices by systematic resampling.
        @param w array of weights
    """
    n = w.shape[0]
    cnw = n * numpy.cumsum(w)
    j = 0
    index = numpy.empty(n, dtype="int")
    for k in xrange(n):
        while cnw[j] < u:
            j = j + 1
        index[k] = j
        u = u + 1.
    return index

def resample_reductive(w, u, n, f_select):
    """ 
        Computes the particle indices and weights by reductive resampling.
        @param w weights
        @param n size of resampled vector
        @param f_select selection algorithm
        @return w resampled weights
        @return index resampled indices
    """
    # select smallest value kappa s.t. sum_j^m min(w_j / kappa, 1) <= n
    kappa = f_select(w.copy(), n)

    # nothing to do
    if kappa is None:
        c = numpy.nan
    else:
        # compute theshold value c s.t. sum_j^m min(c * w_j, 1) = n
        A = (w >= kappa).sum()
        B = (w[numpy.where(w < kappa)[0]]).sum()
        c = (n - A) / B

    # indices of weights to be copied
    if c != c or c == -numpy.inf:
        index = range(n)
        return w[index], index

    index_copy = numpy.where(w * c >= 1)[0]
    index_resample = numpy.where(w * c < 1)[0]

    # number of particles to be resampled
    l = n - index_copy.shape[0]

    # weight to assigned to every index on average
    k = w[index_resample].sum() / l

    # random seed
    u *= k

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


#

# Selection algorithms.

#


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


#

# Test utilities.

#


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

def test_resample(f=resample_reductive, m=2500, n=500, mean=5, sd=5, path='/home/cschafer/Bureau/tmp'):
    """ Tests the resampling algorithm.
        @param f resampling algorithm
        @param m size of weighted sample
        @param n size of resampled system
        @param mean mean of proposal
        @param sd standard deviation of proposal
    """
    w1, x1 = get_importance_weights(m, mean, sd)
    w2, index = f(w1.copy(), numpy.random.random(), n, f_select=select_linear)
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
             n='%d' % n,
             path=path
        )

    f = open('%s.R' % path, 'w')
    f.write(
        '''
        x1=c(%(x1)s)
        w1=c(%(w1)s)
        w1=w1/sum(w1)
        
        x2=c(%(x2)s)
        w2=c(%(w2)s)
        w2=w2/sum(w2)
        
        pdf('%(path)s.pdf', width=12, height=4)
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
    subprocess.Popen(['R', 'CMD', 'BATCH', '--vanilla', '%s.R' % path]).wait()

def main():
    test_resample(m=5000, n=2500, mean=5, sd=5)

if __name__ == "__main__":
    main()
