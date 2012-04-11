#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family obtained via dichotomizing a multivariate Student.
    \namespace binary.student_copula
    \details The correlation structure of the model is limited by the constraints of the elliptic Student copula.
"""

import base
import product
import wrapper

import numpy
import scipy.linalg
import scipy.stats as stats
import time

class StudentCopulaBinary(product.ProductBinary):
    """ Binary parametric family obtained via dichotomizing a multivariate Student. """

    def __init__(self, p, R, delta=None, verbose=False):
        """ 
            Constructor.
            \param p mean
            \param R correlation matrix
        """

        # call super constructor
        super(StudentCopulaBinary, self).__init__(p, name='Student copula family', long_name=__doc__)

        self.py_wrapper = wrapper.student_copula()

        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg', 'binary.Student_copula')

        # add dependent functions
        self.pp_depfuncs += ('_rvslpmf_all',)

        ## target correlation matrix of binary distribution
        self.R = R

        ## target mean vector of binary distribution
        self.p = p
        
        ## degrees of freedom
        self.nu = 5.0

        ## mean vector of auxiliary multivariate Student
        self.mu = stats.t.ppf(self.p, self.nu)

        ## correlation matrix of auxiliary multivariate Student
        self.Q = None

        ## Cholesky decomposition of the correlation matrix of auxiliary multivariate Student
        self.C = None

        # locally adjust correlation matrix of auxiliary multivariate Student
        localQ = calc_local_Q(self.mu, self.p, self.R, delta=delta, nu=self.nu, verbose=verbose)
        # compute the Cholesky decomposition of the locally adjust correlation matrix
        self.C, self.Q = decompose_Q(localQ, mode='scaled', verbose=verbose)


    @classmethod
    def random(cls, d):
        """ 
            Constructs a Student copula model for testing.
            \param cls class 
            \param d dimension
        """
        p, R = base.moments2corr(base.random_moments(d, phi=0.8))
        return cls(p, R)

    def __str__(self):
        return 'mu:\n' + repr(self.mu) + '\nSigma:\n' + repr(self.Q)

    @classmethod
    def _rvs(cls, V, mu, C, nu):
        """ 
            Generates a random variable.
            \param V normal variables
            \param param parameters
            \return binary variables
        """
        d = V.shape[1] - 1
        Y = numpy.empty((V.shape[0], d), dtype=bool)
        for k in xrange(V.shape[0]):
            Y[k] = mu > (numpy.dot(C, V[k, :d]) * numpy.sqrt(nu / V[k, d]))
        return Y

    def _rvsbase(self, size):
        return numpy.hstack((numpy.random.normal(size=(size, self.d)),
                             numpy.random.chisquare(size=(size, 1), df=self.nu)))

    @classmethod
    def independent(cls, p):
        """ 
            Constructs a Student copula model model with independent components.
            \param cls class 
            \param p mean
        """
        return cls(p, numpy.eye(len(p)))

    @classmethod
    def uniform(cls, d):
        """ 
            Construct a uniform distribution as special case of the Student
            copula model
            \param cls class
            \param d dimension
        """
        return cls.independent(p=0.5 * numpy.ones(d))

    @classmethod
    def from_moments(cls, mean, corr, delta=None):
        """ 
            Constructs a Student copula family from given mean and correlation.
            \param mean mean
            \param corr correlation
            \return Student copula family
        """
        return cls(mean.copy(), corr.copy(), delta=delta)

    @classmethod
    def from_data(cls, sample, verbose=False):
        """ 
            Construct a Student copula family from data.
            \param sample a sample of binary data
        """
        return cls(sample.mean, sample.cor, verbose=verbose)

    def renew_from_data(self, sample, lag=0.0, verbose=False, **param):
        """ 
            Re-parameterizes the Student copula family from data.
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

    def getCorr(self):
        """ 
            Computes the correlation matrix induced by the adjusted auxiliary
            multivariate Student correlation matrix Q. The effective
            correlation matrix does not necessarily coincide with the target
            correlation matrix R.
        """
        sd = numpy.sqrt(self.p * (1 - self.p))
        corr = numpy.ones((self.d, self.d))
        for i in xrange(self.d):
            for j in xrange(i):
                corr[i, j] = bvt.cdf([self.mu[i], self.mu[j]], r=self.Q[i, j], nu=3) - self.p[i] * self.p[j]
                corr[i, j] /= (sd[i] * sd[j])
            corr[0:i, i] = corr[i, 0:i].T
        return corr

    corr = property(fget=getCorr, doc="correlation matrix")

    @classmethod
    def test_properties(cls, d, n=1e4, phi=0.8, ncpus=1):
        """
            Tests functionality of the quadratic linear family class.
            \param d dimension
            \param n number of samples
            \param phi dependency level in [0,1]
            \param ncpus number of cpus 
        """

        mean, corr = base.moments2corr(base.random_moments(d, phi=phi))
        print 'given marginals '.ljust(100, '*')
        base.print_moments(mean, corr)

        generator = StudentCopulaBinary.from_moments(mean, corr)
        print generator.name + ':'
        print generator

        print 'exact '.ljust(100, '*')
        base.print_moments(generator.mean, generator.corr)

        #print ('simulation (n = %d) ' % n).ljust(100, '*')
        #binary.base.print_moments(generator.rvs_marginals(n, ncpus))

def calc_local_Q(mu, p, R, nu, eps=0.02, delta=0.005, verbose=False):
    """ 
        Computes the Student correlation matrix Q necessary to generate
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

    k = 0.5 * (sum(R > 0.0) - d)

    for i in range(d):
        for j in range(i):
            localQ[i][j], n = \
                calc_local_q(mu=[mu[i], mu[j]], p=[p[i], p[j]], r=R[i, j], nu=nu, init=R[i, j])
            iterations += n
        localQ[0:i, i] = localQ[i, 0:i].T

    if verbose:
        if k > 0: iterations = float(iterations) / k
        print 'calcLocalQ'.ljust(20) + '> time %.3f, loops %.3f' % (time.time() - t, iterations)
    return localQ


def calc_local_q(mu, p, r, nu, init=0, verbose=False):
    """ 
        Computes the Student bivariate correlation q necessary to generate
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
    q, n = bisectional(mu, p, r, nu=nu, l= -1, u=0, init= -0.5)

    if q == numpy.inf or numpy.isnan(q): q = 0.0
    q = max(min(q, 0.999), -0.999)

    return q, i


def bisectional(mu, p, r, nu, l= -1, u=1, init=0, verbose=False):
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
        v = (bvt.cdf(mu, r=q, nu=3.0) - p[0] * p[1]) / t
        if r < v:
            u = q; q = 0.5 * (q + l)
        else:
            l = q; q = 0.5 * (q + u)
        if abs(l - u) < StudentCopulaBinary.PRECISION: break
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

    for i in xrange(StudentCopulaBinary.MAX_ITERATIONS):
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



#-------------------------------------------------------------- bivariate student t

class _bvt(stats.rv_continuous):
    """
        Bivariate student t distribution with correlation coefficient r.
    """

    def rvs(self, r=0, nu=1.0):
        """
            \param r correlation coefficient
            \param nu degrees of freedom
            \return random bivariate student t
        """
        g = numpy.random.normal(0, 1)
        x = numpy.array([g, r * g + numpy.sqrt(1 - r * r) * numpy.random.normal(0, 1)])
        return x * numpy.sqrt(nu / numpy.random.chisquare(nu))

    def cdf(self, x, r=0, nu=3.0):
        """
            Computes the bivariate Student t cumulative distribution function,
            i.e. the probability that X < x and Y < y. The function only calls lowerDS(x, y, r, nu).
            \param x value
            \param r correlation coefficient
            \param nu degrees of freedom
        """
        return self.lower_DS(nu, x[0], x[1], r=r)

    def ucdf(self, x, nu=None):
        """
            Computes the univariate Student t cumulative distribution function,
            i.e. the probability that X < x.
            \param x value
            \param nu degrees of freedom
        """
        if x == numpy.inf:
            p = 1.0

        elif x == -numpy.inf:
            p = 0.0

        elif nu < 1:
            p = _bvt.phid(x)

        elif nu == 1:
            p = (1 + 2 * numpy.arctan(x) / numpy.pi) / 2.0

        elif nu == 2:
            p = (1 + x / numpy.sqrt(2 + x * x)) / 2.0

        else:
            tt = x * x
            cssthe = 1 / (1 + tt / nu)
            polyn = 1

            for j in range(nu - 2, 1, -2): # ??
                polyn = 1 + (j - 1) * cssthe * polyn / j
            if nu % 2 == 1:
                rn = nu
                ts = x / numpy.sqrt(rn)

                p = (1 + 2 * (numpy.arctan(ts) + ts * cssthe * polyn) / numpy.pi) / 2.0
            else:
                snthe = x / numpy.sqrt(nu + tt)
                p = (1 + snthe * polyn) / 2.0

            p = max(0, min(p))

        return p

    def lower_DS(self, nu=None, dh=None, dk=None, r=None, eps=1e-10):
        """
            A function for computing bivariate t probabilities.
            bvtl calculates the probability that x < dh and y < dk
           
            \param nu integer number of degrees of freedom, nu < 1, gives normal case
            \param dh 1st upper integration limit
            \param dk 2nd upper integration limit
            \param r   correlation coefficient

            This function is based on the method described by Dunnett, C.W. and M. Sobel, (1954),
            A bivariate generalization of Student's t-distribution with tables for certain special cases,
            Biometrika 41, pp. 153-169.
            
            The code was adapted for python from the matlab routine by Alan Genz.
        """
        if nu < 1:
            raise ValueError

        elif dh == -numpy.inf or dk == -numpy.inf:
            p = 0

        elif dh == numpy.inf:
            if dk == numpy.inf:
                p = 1
            else:
                p = _bvt.studnt(nu, dk)

        elif dk == numpy.inf:
            p = _bvt.studnt(nu, dh)

        elif 1 - r < eps:
            p = _bvt.studnt(nu, min(dh, dk))

        elif r + 1 < eps:
            p = 0

            if dh > -dk:
                p = _bvt.studnt(nu, dh) - _bvt.studnt(nu, -dk)
        else:
            tpi = 2 * numpy.pi
            ors = 1 - r * r
            hrk = dh - r * dk
            krh = dk - r * dh

            if abs(hrk) + ors > 0:
                xnhk = hrk ** 2 / (hrk ** 2 + ors * (nu + dk ** 2))
                xnkh = krh ** 2 / (krh ** 2 + ors * (nu + dh ** 2))
            else:
                xnhk = 0; print xnhk
                xnkh = 0

            hs = numpy.sign(dh - r * dk)
            ks = numpy.sign(dk - r * dh)

            if nu % 2 == 0:
                bvt = numpy.arctan2(numpy.sqrt(ors), -r) / tpi
                gmph = dh / numpy.sqrt(16 * (nu + dh ** 2))
                gmpk = dk / numpy.sqrt(16 * (nu + dk ** 2))

                btnckh = 2 * numpy.arctan2(numpy.sqrt(xnkh), numpy.sqrt(1 - xnkh)) / numpy.pi
                btpdkh = 2 * numpy.sqrt(xnkh * (1 - xnkh)) / numpy.pi
                btnchk = 2 * numpy.arctan2(numpy.sqrt(xnhk), numpy.sqrt(1 - xnhk)) / numpy.pi
                btpdhk = 2 * numpy.sqrt(xnhk * (1 - xnhk)) / numpy.pi
                for j in xrange(1, int(nu) / 2 + 1):
                    bvt = bvt + gmph * (1 + ks * btnckh)
                    bvt = bvt + gmpk * (1 + hs * btnchk)
                    btnckh = btnckh + btpdkh
                    btpdkh = 2 * j * btpdkh * (1 - xnkh) / (2 * j + 1)

                    btnchk = btnchk + btpdhk
                    btpdhk = 2 * j * btpdhk * (1 - xnhk) / (2 * j + 1)

                    gmph = gmph * (j - 1 / 2) / (j * (1 + dh ** 2 / nu))
                    gmpk = gmpk * (j - 1 / 2) / (j * (1 + dk ** 2 / nu))

            else:
                qhrk = numpy.sqrt(dh ** 2 + dk ** 2 - 2 * r * dh * dk + nu * ors)

                hkrn = dh * dk + r * nu
                hkn = dh * dk - nu
                hpk = dh + dk

                bvt = numpy.arctan2(-numpy.sqrt(nu) * (hkn * qhrk + hpk * hkrn), hkn * hkrn - nu * hpk * qhrk) / tpi
                if bvt < -10 * eps:
                    bvt = bvt + 1

                gmph = dh / (tpi * numpy.sqrt(nu) * (1 + dh ** 2 / nu))
                gmpk = dk / (tpi * numpy.sqrt(nu) * (1 + dk ** 2 / nu))
                btnckh = numpy.sqrt(xnkh)
                btpdkh = btnckh

                btnchk = numpy.sqrt(xnhk)
                btpdhk = btnchk

                for j in xrange(1, (int(nu) - 1) / 2 + 1):
                    bvt = bvt + gmph * (1 + ks * btnckh)
                    bvt = bvt + gmpk * (1 + hs * btnchk)
                    btpdkh = (2 * j - 1) * btpdkh * (1 - xnkh) / (2 * j)
                    btnckh = btnckh + btpdkh

                    btpdhk = (2 * j - 1) * btpdhk * (1 - xnhk) / (2 * j)
                    btnchk = btnchk + btpdhk

                    gmph = gmph * j / ((j + 1 / 2) * (1 + dh ** 2 / nu))
                    gmpk = gmpk * j / ((j + 1 / 2) * (1 + dk ** 2 / nu))
            p = bvt
        return p

bvt = _bvt(name='bvt', longname='A bivariate t-distribution', shapes='r')
