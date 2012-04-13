#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family obtained via dichotomizing a multivariate auxiliary distribution.
    \namespace binary.copula
    \details The correlation structure of the model is limited by the constraints of the elliptic auxiliary copula.
"""

import binary.base as base
import binary.product as product

import numpy
import scipy.linalg
import scipy.stats as stats
import time

class CopulaBinary(product.ProductBinary):
    """ Binary parametric family obtained via dichotomizing a multivariate auxiliary distribution. """

    def __init__(self, p, R, delta=None, name='Copula family', long_name=__doc__):
        """ 
            Constructor.
            \param p mean
            \param R correlation matrix
        """

        # call super constructor
        super(CopulaBinary, self).__init__(p, name=name, long_name=long_name)

        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg',)

        # add dependent functions
        self.pp_depfuncs += ('_rvslpmf_all',)

        ## target correlation matrix of binary distribution
        self.R = R

        ## target mean vector of binary distribution
        self.p = p

        ## mean vector of multivariate auxiliary distribution
        self.mu = self.aux_ppf(self.p)

        ## correlation matrix of multivariate auxiliary distribution
        self.Q = None

        ## Cholesky decomposition of the correlation matrix of multivariate auxiliary distribution
        self.C = None

        # locally adjust correlation matrix of multivariate auxiliary distribution
        localQ = self.calc_local_Q(self.mu, self.p, self.R, delta=delta, verbose=False)

        # compute the Cholesky decomposition of the locally adjust correlation matrix
        self.C, self.Q = decompose_Q(localQ, mode='scaled', verbose=False)

    def __str__(self):
        return 'mu:\n' + repr(self.mu) + '\nSigma:\n' + repr(self.Q)

    @classmethod
    def random(cls, d):
        """ 
            Constructs a auxiliary copula model for testing.
            \param cls class 
            \param d dimension
        """
        p, R = base.moments2corr(base.random_moments(d, phi=0.8))
        return cls(p, R)

    @classmethod
    def independent(cls, p):
        """ 
            Constructs a auxiliary copula model model with independent components.
            \param cls class 
            \param p mean
        """
        return cls(p, numpy.eye(len(p)))

    @classmethod
    def uniform(cls, d):
        """ 
            Construct a uniform distribution.
            copula model
            \param cls class
            \param d dimension
        """
        return cls.independent(p=0.5 * numpy.ones(d))

    @classmethod
    def from_moments(cls, mean, corr, delta=None):
        """ 
            Constructs a gaussian copula family from given mean and correlation.
            \param mean mean
            \param corr correlation
            \return gaussian copula family
        """
        return cls(mean.copy(), corr.copy(), delta=delta)

    @classmethod
    def from_data(cls, sample, verbose=False):
        """ 
            Construct a auxiliary copula family from data.
            \param sample a sample of binary data
        """
        return cls(sample.mean, sample.cor, verbose=verbose)

    def renew_from_data(self, sample, lag=0.0, verbose=False, **param):
        """ 
            Re-parameterizes the auxiliary copula family from data.
            \param sample a sample of binary data
            \param lag update lag
            \param verbose verbose     
            \param param parameters
        """
        self.R = sample.getCor(weight=True)
        newP = sample.getMean(weight=True)
        self.p = (1 - lag) * newP + lag * self.p

        ## mean of hidden stats.normal distribution
        self.mu = stats.norm.ppf(self.p)

        localQ = self.calc_local_Q(self.mu, self.p, self.R, verbose=verbose)

        ## correlation matrix of the hidden stats.normal distribution
        self.C, self.Q = decompose_Q(localQ, mode='scaled', verbose=verbose)

    def exact_marginals(self, ncpus=None):
        """ 
            Computes the correlation matrix induced by the adjusted auxiliary
            multivariate distribution correlation matrix Q. The effective
            correlation matrix does not necessarily coincide with the target
            correlation matrix R.
        """
        sd = numpy.sqrt(self.p * (1 - self.p))
        corr = numpy.ones((self.d, self.d))
        for i in xrange(self.d):
            for j in xrange(i):
                corr[i, j] = self.aux_cdf([self.mu[i], self.mu[j]], self.Q[i, j]) - self.p[i] * self.p[j]
                corr[i, j] /= (sd[i] * sd[j])
            corr[0:i, i] = corr[i, 0:i].T
        return self.mean, corr

    def getCorr(self):
        return self.exact_marginals()[1]

    corr = property(fget=getCorr, doc="correlation matrix")

    @classmethod
    def calc_local_Q(cls, mu, p, R, eps=0.02, delta=0.005, verbose=False):
        """ 
            Computes the auxiliary correlation matrix Q necessary to generate
            bivariate Bernoulli samples with a certain local correlation matrix R.
            \param R correlation matrix of the binary
            \param mu mean of the hidden stats.normal
            \param p mean of the binary
            \param verbose print to stdout 
        """
        t = time.time()
        d = len(p)

        iterations = 0
        localQ = numpy.ones((d, d))

        for i in range(d):
            if p[i] < eps or p[i] > 1.0 - eps:
                R[i, :] = R[:, i] = numpy.zeros(d, dtype=float)
                R[i, i] = 1.0
            else:
                for j in range(i):
                    if abs(R[i, j]) < delta:
                        R[i, j] = 0.0
            R[:i, i] = R[i, :i]

        for i in range(d):
            for j in range(i):
                localQ[i][j], n = \
                    cls.calc_local_q(mu=[mu[i], mu[j]], p=[p[i], p[j]], r=R[i][j], init=R[i][j])
                iterations += n
            localQ[0:i, i] = localQ[i, 0:i].T

        if verbose: print 'calcLocalQ'.ljust(20) + '> time %.3f, loops %.3f' % (time.time() - t, iterations)
        return localQ

    @classmethod
    def calc_local_q(cls, mu, p, r, init=0):
        """ 
            Computes the auxiliary bivariate correlation q necessary to generate
            bivariate Bernoulli samples with a given correlation r.
            \param mu mean of the hidden stats.normal
            \param p mean of the binary            
            \param r correlation between the binary
            \param init initial value
            \param verbose print to stdout 
        """

        if r == 0.0: return 0.0, 0

        # For extreme marginals, correlation is negligible.
        if cls.NU == 1.0:
            for i in range(2):
                if abs(mu[i]) > 3.2:
                    return 0.0, 0

        # Ensure r is feasible.
        t = numpy.sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
        maxr = min(0.999, (min(p[0], p[1]) - p[0] * p[1]) / t)
        minr = max(-0.999, (max(p[0] + p[1] - 1, 0) - p[0] * p[1]) / t)
        r = min(max(r, minr), maxr)

        # Solve implicit form by iteration.
        q, i = cls.newtonraphson(mu, p, r, init)
        if q < -1.0:
            q, i = cls.bisectional(mu, p, r)

        if q == numpy.inf or numpy.isnan(q): q = 0.0
        q = max(min(q, 0.999), -0.999)

        return q, i

    @classmethod
    def newtonraphson(cls, mu, p, r, init=0, verbose=False):
        return -numpy.inf, 0.0

    @classmethod
    def bisectional(cls, mu, p, r, l= -1.0, u=1.0, init=0.0, verbose=False):
        """
            Bisectional search for the correlation parameter q of the underlying normal distribution.
            \param mu mean of the hidden stats.normal
            \param p mean of the binary            
            \param r correlation between the binary
            \param l lower bound
            \param u upper bound         
            \param init initial value
            \param verbose print to stdout 
        """
        if verbose: print '\nBisectional search.'

        t = numpy.sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
        q = init

        N = 50
        for i in xrange(N):
            if verbose: print q
            v = (cls.aux_cdf(mu, q) - p[0] * p[1]) / t
            if r < v:
                u = q; q = 0.5 * (q + l)
            else:
                l = q; q = 0.5 * (q + u)
            if abs(l - u) < CopulaBinary.PRECISION: break

        return q, i

    @classmethod
    def aux_cdf(cls, x, r):
        pass

    @classmethod
    def aux_ppf(cls, p):
        pass


def decompose_Q(Q, mode='scaled', verbose=False):
    """
        Computes the Cholesky decomposition of Q. If Q is not positive definite, either the
        identity matrix, a scaled version of Q or the correlation matrix nearest to Q is used.
        \param Q symmetric matrix 
        \param mode way of dealing with non-definite matrices [independent, scaled, nearest] 
        \param verbose print to stdout 
    """
    t = time.time()
    d = Q.shape[0]
    try:
        C = scipy.linalg.cholesky(Q, True)
    except (scipy.linalg.LinAlgError, ValueError):
        if mode == 'independent':
            return numpy.eye(d), numpy.eye(d)
        if mode == 'scaled':
            Q = scale_Q(Q, verbose=verbose)
        if mode == 'nearest':
            Q = nearest_Q(Q, verbose=verbose)

    try:
        C = scipy.linalg.cholesky(Q, True)
    except scipy.linalg.LinAlgError:
        print "WARNING: Set matrix to identity."
        C, Q = numpy.eye(d), numpy.eye(d)

    if verbose: print 'decomposeQ'.ljust(20) + '> time %.3f' % (time.time() - t)

    return C, Q

def scale_Q(Q, verbose=False):
    """
        Rescales the locally adjusted matrix Q to make it positive definite.
        \param Q symmetric matrix 
        \param verbose print to stdout 
    """
    t = time.time()
    d = Q.shape[0]
    try:
        n = abs(Q).sum() - d
    except:
        print "WARNING: Could not evaluate stats.norm."
        n = 1.0

    # If the smallest eigenvalue is (almost) negative, re-scale Q matrix.
    mineig = min(scipy.linalg.eigvalsh(Q)) - 1e-04
    if mineig < 0:
        Q -= mineig * numpy.eye(d)
        Q /= (1 - mineig)

    if verbose: print 'scaleQ'.ljust(20) + '> time %.3f, ratio %.3f' % (time.time() - t, (abs(Q).sum() - d) / n)

    return Q

def nearest_Q(Q, verbose=False):
    """
        Computes the nearest (Frobenius stats.norm) correlation matrix for the
        locally adjusted matrix Q. The nearest correlation matrix problem is
        solved using the alternating projection method proposed in <i>Computing
        the Nearest Correlation Matrix - A problem from Finance</i> by N. Higham
        (2001).
        
        \param Q symmetric matrix 
        \param verbose print to stdout
    """
    t = time.time()
    d = len(Q[0])
    n = abs(Q).sum() - d

    # run alternating projections
    S = numpy.zeros((d, d))
    Y = Q

    for i in xrange(CopulaBinary.MAX_ITERATIONS):
        # Dykstra's correction term
        R = Y - S

        # Project corrected Y matrix on convex set of positive definite matrices.
        D, E = scipy.linalg.eigh(R)
        for j in range(len(D)):
            D[j] = max(D[j], numpy.exp(-8.0))
        X = numpy.dot(numpy.dot(E, numpy.diag(D)), E.T)

        # Update correction term.
        S = X - R

        # Project X matrix on convex set of matrices with unit numpy.diagonal.
        Y = X.copy()
        for j in range(len(D)): Y[j][j] = 1.

        if scipy.linalg.norm(X - Y) < 0.001: break

    q = numpy.diagonal(X)[numpy.newaxis, :]
    Q = X / numpy.sqrt(numpy.dot(q.T, q))
    if verbose: print 'nearestQ'.ljust(20) + '> time %.3f, loops %i, ratio %.3f' % (time.time() - t, i, (abs(Q).sum() - d) / n)

    return Q
