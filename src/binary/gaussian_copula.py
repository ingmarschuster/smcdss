#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family obtained via dichotomizing a multivariate Gaussian. """

"""
\namespace binary.gaussian_copula
$Author: christian.a.schafer@gmail.com $
$Rev: 144 $
$Date: 2011-05-12 19:12:23 +0200 (Do, 12 Mai 2011) $
\details The correlation structure of the model is limited by the constraints of the elliptic Gaussian copula.
"""

import binary.base
import binary.product
import binary.wrapper
import numpy
import scipy.linalg
import scipy.stats as stats
import time


class GaussianCopulaBinary(binary.product.ProductBinary):
    """ Binary parametric family obtained via dichotomizing a multivariate Gaussian. """

    def __init__(self, p, R, verbose=False):
        """ 
            Constructor.
            \param p mean
            \param R correlation matrix
        """

        # call super constructor
        binary.product.ProductBinary.__init__(self, p, name='Gaussian copula family', long_name=__doc__)

        self.py_wrapper = binary.wrapper.gaussian_copula()
        
        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg', 'binary.gaussian_copula')

        # add dependent functions
        self.pp_depfuncs += ('_rvslpmf_all',)

        ## target correlation matrix of binary distribution
        self.R = R

        ## target mean vector of binary distribution
        self.p = p

        ## mean vector of auxiliary multivariate Gaussian
        self.mu = stats.norm.ppf(self.p)

        ## correlation matrix of auxiliary multivariate Gaussian
        self.Q = None

        ## Cholesky decomposition of the correlation matrix of auxiliary multivariate Gaussian
        self.C = None

        # locally adjust correlation matrix of auxiliary multivariate Gaussian
        localQ = calc_local_Q(self.mu, self.p, self.R, verbose=verbose)
        # compute the Cholesky decomposition of the locally adjust correlation matrix
        self.C, self.Q = decompose_Q(localQ, mode='scaled', verbose=verbose)


    @classmethod
    def random(cls, d):
        """ 
            Constructs a Gaussian copula model for testing.
            \param cls class 
            \param d dimension
        """
        p, R = binary.base.moments2corr(binary.base.random_moments(d, phi=0.8))
        return cls(p, R)

    def __str__(self):
        return 'mu:\n' + repr(self.mu) + '\nSigma:\n' + repr(self.Q)

    @classmethod
    def _rvs(cls, V, mu, C):
        """ 
            Generates a random variable.
            \param V normal variables
            \param param parameters
            \return binary variables
        """
        Y = numpy.empty((V.shape[0], V.shape[1]), dtype=bool)
        for k in xrange(V.shape[0]):
            Y[k] = mu > numpy.dot(C, V[k])
        return Y

    @classmethod
    def independent(cls, p):
        """ 
            Constructs a Gaussian copula model model with independent components.
            \param cls class 
            \param p mean
        """
        return cls(p, numpy.eye(len(p)))

    @classmethod
    def uniform(cls, d):
        """ 
            Construct a uniform distribution as special case of the Gaussian
            copula model
            \param cls class
            \param d dimension
        """
        return cls.independent(p=0.5 * numpy.ones(d))

    @classmethod
    def from_moments(cls, mean, corr):
        """ 
            Constructs a gaussian copula family from given mean and correlation.
            \param mean mean
            \param corr correlation
            \return gaussian copula family
        """
        return cls(mean, corr)

    @classmethod
    def from_data(cls, sample, verbose=False):
        """ 
            Construct a Gaussian copula family from data.
            \param sample a sample of binary data
        """
        return cls(sample.mean, sample.cor, verbose=verbose)

    def renew_from_data(self, sample, lag=0.0, verbose=False, **param):
        """ 
            Re-parameterizes the Gaussian copula family from data.
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

        localQ = calc_local_Q(self.mu, self.p, self.R, verbose=verbose)

        ## correlation matrix of the hidden stats.normal distribution
        self.C, self.Q = decompose_Q(localQ, mode='scaled', verbose=verbose)

    def _rvsbase(self, size):
        return numpy.random.normal(size=(size, self.d))


    def getCorr(self):
        """ 
            Computes the correlation matrix induced by the adjusted auxiliary
            multivariate Gaussian correlation matrix Q. The effective
            correlation matrix does not necessarily coincide with the target
            correlation matrix R.
        """
        sd = numpy.sqrt(self.p * (1 - self.p))
        corr = numpy.ones((self.d, self.d))
        for i in xrange(self.d):
            for j in xrange(i):
                corr[i, j] = bvnorm.cdf([self.mu[i], self.mu[j]], self.Q[i, j]) - self.p[i] * self.p[j]
                corr[i, j] /= (sd[i] * sd[j])
            corr[0:i, i] = corr[i, 0:i].T
        return corr

    corr = property(fget=getCorr, doc="correlation matrix")

    @classmethod
    def test_properties(cls,d, n=1e4, phi=0.8, ncpus=1):
        """
            Tests functionality of the quadratic linear family class.
            \param d dimension
            \param n number of samples
            \param phi dependency level in [0,1]
            \param ncpus number of cpus 
        """

        mean, corr = binary.base.moments2corr(binary.base.random_moments(d, phi=phi))
        print 'given marginals '.ljust(100, '*')
        binary.base.print_moments(mean, corr)

        generator = GaussianCopulaBinary.from_moments(mean, corr)
        print generator.name + ':'
        print generator

        print 'exact '.ljust(100, '*')
        binary.base.print_moments(generator.mean, generator.corr)

        print ('simulation (n = %d) ' % n).ljust(100, '*')
        binary.base.print_moments(generator.rvs_marginals(n, ncpus))

def calc_local_Q(mu, p, R, eps=0.02, delta=0.075, verbose=False):
    """ 
        Computes the Gaussian correlation matrix Q necessary to generate
        bivariate Bernoulli samples with a certain local correlation matrix R.
        \param R correlation matrix of the binary
        \param mu mean of the hidden stats.normal
        \param p mean of the binary
        \param verbose print to stdout 
    """

    t = time.time()
    iterations = 0
    d = len(p)
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
    k = 0.5 * (sum(R > 0.0) - d)

    for i in range(d):
        for j in range(i):
            localQ[i][j], n = \
                calc_local_q(mu=[mu[i], mu[j]], p=[p[i], p[j]], r=R[i][j], init=R[i][j])
            iterations += n
        localQ[0:i, i] = localQ[i, 0:i].T

    if verbose:
        if k > 0: iterations = float(iterations) / k
        print 'calcLocalQ'.ljust(20) + '> time %.3f, loops %.3f' % (time.time() - t, iterations)
    return localQ


def calc_local_q(mu, p, r, init=0, verbose=False):
    """ 
        Computes the Gaussian bivariate correlation q necessary to generate
        bivariate Bernoulli samples with a given correlation r.
        \param mu mean of the hidden stats.normal
        \param p mean of the binary            
        \param r correlation between the binary
        \param init initial value
        \param verbose print to stdout 
    """

    if r == 0.0: return 0.0, 0

    # For extreme marginals, correlation is negligible.
    for i in range(2):
        if abs(mu[i]) > 3.2: return 0.0, 0

    # Ensure r is feasible.
    t = numpy.sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
    maxr = min(0.999, (min(p[0], p[1]) - p[0] * p[1]) / t)
    minr = max(-0.999, (max(p[0] + p[1] - 1, 0) - p[0] * p[1]) / t)
    r = min(max(r, minr), maxr)

    # Solve implicit form by iteration.
    q, i = newtonraphson(mu, p, r, init)
    if q < -1:
        q, n = bisectional(mu, p, r, l= -1, u=0, init= -0.5)
        i += n

    if q == numpy.inf or numpy.isnan(q): q = 0
    q = max(min(q, 0.999), -0.999)

    return q, i


def newtonraphson(mu, p, r, init=0, verbose=False):
    """
        Newton-Raphson search for the correlation parameter q of the underlying normal distribution.
        \param mu mean of the hidden stats.normal
        \param p mean of the binary            
        \param r correlation between the binary
        \param init initial value
        \param verbose print to stdout 
    """
    if verbose: print '\nNewton-Raphson search.'

    t = numpy.sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
    s = p[0] * p[1] + r * t

    greater_one = False
    q = init
    last_q = numpy.inf

    for i in xrange(GaussianCopulaBinary.MAX_ITERATIONS):
        q = q - round((bvnorm.cdf(mu, q) - s), 8) / bvnorm.pdf(mu, q)
        if verbose: print q
        if q > 1:
            q = 0.999
            if greater_one == True:
                break                  # avoid endless loop
            else:
                greater_one = True     # restart once at boundary

        if q < -1:
            break

        if abs(last_q - q) < GaussianCopulaBinary.PRECISION: break
        last_q = q

    return q, i


def bisectional(mu, p, r, l= -1, u=1, init=0, verbose=False):
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
        v = (bvnorm.cdf(mu, q) - p[0] * p[1]) / t
        if r < v:
            u = q; q = 0.5 * (q + l)
        else:
            l = q; q = 0.5 * (q + u)
        if abs(l - u) < GaussianCopulaBinary.PRECISION: break

    return q, i


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

    # If the smallest eigenvalue is (almost) negative, rescale Q matrix.
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

    for i in xrange(GaussianCopulaBinary.MAX_ITERATIONS):
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



#-------------------------------------------------------------- bivariate normal

class _bvnorm(stats.rv_continuous):
    """
        bivariate normal distribution with correlation coefficient r.
        pdf(x,y) = numpy.exp(-(x*x-2*r*x*y+y*y)/(2*(1-r*r))) / (2*numpy.pi*numpy.sqrt(1-r*r))
    """

    def cdf(self, x, r=0):
        """
            Computes the bivariate normal cumulative distribution function,
            i.e. the probability that X < x and Y < y. The function only calls lowerDW(x, y, r).
            \param x value
            \param r correlation coefficient 
        """
        return self.lower_DW(x[0], x[1], r)

    def pdf(self, x, r=0):
        """
            Computes the bivariate normal probability distribution function, i.e. the density at (x, y)
            \param x value
            \param r correlation coefficient 
        """
        z = x[0] * x[0] - 2 * r * x[0] * x[1] + x[1] * x[1]
        return numpy.exp(-z / (2 * (1 - r * r))) / (2 * numpy.pi * numpy.sqrt(1 - r * r))

    def rvs(self, r=0):
        """
            \param r correlation coefficient 
            \return random bivariate normal
        """
        v = numpy.random.normal(0, 1)
        return r * v + numpy.sqrt(1 - r * r) * numpy.random.normal(0, 1)

    def lower_DW(self, dh, dk, r):
        """
            Computes bivariate normal probabilities; lowerDW calculates the probability
            that x < dh and y < dk using the Drezner-Wesolowsky approximation.
            The function only calls upperDW(-dh, -dk, r).
            
            \param dh 1st lower integration limit
            \param dk 2nd lower integration limit
            \param r correlation coefficient
        """
        return self.upper_DW(-dh, -dk, r)

    def upper_DW(self, dh, dk, r):
        """
            Computes bivariate normal probabilities; upperDW calculates the probability that x > dh and y > dk. 
              
            This function is based on the method described by Z. Drezner and G.O. Wesolowsky, (1989),
            "On the computation of the bivariate normal integral", Journal of Statist.
            Comput. Simul. 35, pp. 101-107, with major modifications for double precision, for |r| close to 1.
        
            The code was adapted for python from the matlab routine by Alan Genz.
            
            \param dh 1st lower integration limit
            \param dk 2nd lower integration limit
            \param r correlation coefficient
        """
        twopi = 2 * numpy.pi
        if abs(r) < 0.3:
            lg = 3
        #       Gauss Legendre points and weights, n =  6
            w = [0.1713244923791705, 0.3607615730481384, 0.4679139345726904]
            x = [0.9324695142031522, 0.6612093864662647, 0.2386191860831970]
        elif abs(r) < 0.75:
            lg = 6;
        #       Gauss Legendre points and weights, n = 12
            w = [.04717533638651177, 0.1069393259953183, 0.1600783285433464, \
                 0.2031674267230659, 0.2334925365383547, 0.2491470458134029]
            x = [0.9815606342467191, 0.9041172563704750, 0.7699026741943050, \
                 0.5873179542866171, 0.3678314989981802, 0.1252334085114692]
        else:
            lg = 10
        #       Gauss Legendre points and weights, n = 20
            w = [.01761400713915212, .04060142980038694, .06267204833410906, \
                       .08327674157670475, 0.1019301198172404, 0.1181945319615184, \
                       0.1316886384491766, 0.1420961093183821, 0.1491729864726037, \
                       0.1527533871307259]
            x = [0.9931285991850949, 0.9639719272779138, 0.9122344282513259, \
                       0.8391169718222188, 0.7463319064601508, 0.6360536807265150, \
                       0.5108670019508271, 0.3737060887154196, 0.2277858511416451, \
                       0.07652652113349733]

        h = dh; k = dk; hk = h * k; bvn = 0
        if abs(r) < 0.925:
            hs = (h * h + k * k) / 2; asr = numpy.arcsin(r);
            for i in range(lg):
                sn = numpy.sin(asr * (1 - x[i]) / 2);
                bvn = bvn + w[i] * numpy.exp((sn * hk - hs) / (1 - sn * sn));
                sn = numpy.sin(asr * (1 + x[i]) / 2);
                bvn = bvn + w[i] * numpy.exp((sn * hk - hs) / (1 - sn * sn));
            bvn = bvn * asr / (4 * numpy.pi) + stats.norm.cdf(-h) * stats.norm.cdf(-k)
        else:
            if r < 0:
                    k = -k
                    hk = -hk
            if abs(r) < 1:
                aas = (1 - r) * (1 + r); a = numpy.sqrt(aas); bs = (h - k) ** 2;
                c = (4 - hk) / 8 ; d = (12 - hk) / 16; asr = -(bs / aas + hk) / 2;
                if asr > -100:
                    bvn = a * numpy.exp(asr) * (1 - c * (bs - aas) * (1 - d * bs / 5) / 3 + c * d * aas * aas / 5);
                if -hk < 100:
                    b = numpy.sqrt(bs); sp = numpy.sqrt(twopi) * stats.norm.cdf(-b / a);
                    bvn = bvn - numpy.exp(-hk / 2) * sp * b * (1 - c * bs * (1 - d * bs / 5) / 3);
                a = a / 2;
                for i in range(lg):
                    for iis in range(-1 , 3 , 2):
                        xs = (a * (iis * x[i] + 1)) ** 2; rs = numpy.sqrt(1 - xs);
                        asr = -(bs / xs + hk) / 2;
                        if asr > -100:
                            sp = (1 + c * xs * (1 + d * xs));
                            ep = numpy.exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs;
                            bvn = bvn + a * w[i] * numpy.exp(asr) * (ep - sp);
                bvn = -bvn / twopi;
            if r > 0: bvn = bvn + stats.norm.cdf(-max(h, k))
            if r < 0: bvn = -bvn + max(0, stats.norm.cdf(-h) - stats.norm.cdf(-k))
        p = max(0, min(1, bvn));
        return p

    def lower_MC(self, dh, dk, r, n=100000):
        """
            Computes bivariate stats.normal probabilities; lowerMC calculates the probability that x < dh and y < dk
            using a Monte Carlo approximation of n samples. The function only calls upperMC(-dh, -dk, r, n).

            \param dh 1st lower integration limit
            \param dk 2nd lower integration limit
            \param r   correlation coefficient
            \param n   sample size
        """
        return self.upper_MC(-dh, -dk, r, n)

    def upper_MC(self, dh, dk, r, n=100000):
        """
            Computes bivariate normal probabilities; upperMC calculates the probability that x > dh and y > dk. 
            This function is a simple MC evaluation used to cross-check the DW approximation algorithms.
        
            \param dh 1st lower integration limit
            \param dk 2nd lower integration limit
            \param r   correlation coefficient
            \param n   sample size
        
        """
        p = 0
        for i in range(n):
            v1 = numpy.random.normal(0, 1)
            v2 = r * v1 + numpy.sqrt(1 - r * r) * numpy.random.normal(0, 1)
            if v1 > dh and v2 > dk:p += 1
        return p / float(n)

bvnorm = _bvnorm(name='bvnorm', longname='A bivariate normal', shapes='r')
