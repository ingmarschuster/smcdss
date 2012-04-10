#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family obtained via dichotomizing a multivariate Student. """

"""
\namespace binary.student_copula
$Author: christian.a.schafer@gmail.com $
$Rev: 144 $
$Date: 2011-05-12 19:12:23 +0200 (Do, 12 Mai 2011) $
\details The correlation structure of the model is limited by the constraints of the elliptic Student copula.
"""

import binary.base
import binary.product
import binary.wrapper
import numpy
import scipy.linalg
import scipy.stats as stats
import time

class StudentCopulaBinary(binary.product.ProductBinary):
    """ Binary parametric family obtained via dichotomizing a multivariate Student. """

    def __init__(self, p, R, delta=None, verbose=False):
        """ 
            Constructor.
            \param p mean
            \param R correlation matrix
        """

        # call super constructor
        binary.product.ProductBinary.__init__(self, p, name='Student copula family', long_name=__doc__)

        self.py_wrapper = binary.wrapper.student_copula()

        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg', 'binary.Student_copula')

        # add dependent functions
        self.pp_depfuncs += ('_rvslpmf_all',)

        ## target correlation matrix of binary distribution
        self.R = R

        ## target mean vector of binary distribution
        self.p = p

        ## mean vector of auxiliary multivariate Student
        self.mu = stats.t.ppf(self.p, 3)
        #x=0.5; print t.ppf(t.cdf(x, 3), 3)

        ## correlation matrix of auxiliary multivariate Student
        self.Q = None

        ## Cholesky decomposition of the correlation matrix of auxiliary multivariate Student
        self.C = None

        # locally adjust correlation matrix of auxiliary multivariate Student
        localQ = calc_local_Q(self.mu, self.p, self.R, delta=delta, verbose=verbose)
        # compute the Cholesky decomposition of the locally adjust correlation matrix
        self.C, self.Q = decompose_Q(localQ, mode='scaled', verbose=verbose)


    @classmethod
    def random(cls, d):
        """ 
            Constructs a Student copula model for testing.
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

    def _rvsbase(self, size):
        return numpy.random.normal(size=(size, self.d))


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

        mean, corr = binary.base.moments2corr(binary.base.random_moments(d, phi=phi))
        print 'given marginals '.ljust(100, '*')
        binary.base.print_moments(mean, corr)

        generator = StudentCopulaBinary.from_moments(mean, corr)
        print generator.name + ':'
        print generator

        print 'exact '.ljust(100, '*')
        binary.base.print_moments(generator.mean, generator.corr)

        #print ('simulation (n = %d) ' % n).ljust(100, '*')
        #binary.base.print_moments(generator.rvs_marginals(n, ncpus))

def calc_local_Q(mu, p, R, eps=0.02, delta=None, verbose=False):
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

    if delta is None:
        delta = 2.0 * scipy.linalg.norm(numpy.tril(R, k= -1) + numpy.triu(R, k=1)) / float((d - 1) * d)

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
                calc_local_q(mu=[mu[i], mu[j]], p=[p[i], p[j]], r=R[i][j], init=R[i][j])
            iterations += n
        localQ[0:i, i] = localQ[i, 0:i].T

    if verbose:
        if k > 0: iterations = float(iterations) / k
        print 'calcLocalQ'.ljust(20) + '> time %.3f, loops %.3f' % (time.time() - t, iterations)
    return localQ


def calc_local_q(mu, p, r, init=0, verbose=False):
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
    q, n = bisectional(mu, p, r, l= -1, u=0, init= -0.5)

    if q == numpy.inf or numpy.isnan(q): q = 0.0
    q = max(min(q, 0.999), -0.999)

    return q, i

'''
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

    for i in xrange(StudentCopulaBinary.MAX_ITERATIONS):
        try:
            q = q - round((bvt.cdf(mu, r=q, nu=3) - s), 8) / bvnorm.pdf(mu, q)
        except FloatingPointError:
            return 0.0, i
        if verbose: print q
        if q > 1:
            q = 0.999
            if greater_one == True:
                break                  # avoid #endless loop
            else:
                greater_one = True     # restart once at boundary

        if q < -1:
            break

        if abs(last_q - q) < StudentCopulaBinary.PRECISION: break
        last_q = q

    return q, i
'''


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
        v = (bvt.cdf(mu, q) - p[0] * p[1]) / t
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



#-------------------------------------------------------------- bivariate normal

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

nu = 4.0
r = 0.25

l = numpy.zeros((2, 2))
m = numpy.zeros(2)
n = 1e6
mu = [-0.25, 0.5]
p = 0
for i in xrange(int(n)):
    x = bvt.rvs(r, nu)
    l += numpy.outer(x, x)
    m += x
    if (x < mu).all():
        p += 1.0
m = m / n
#print m
v = l / n - numpy.outer(m, m)
#print v
v = v * nu / (nu - 2)
print v / numpy.sqrt(numpy.outer(numpy.diag(v), numpy.diag(v)))
print 'MC p: ', p / n
print bvt.cdf(mu, r, nu)

'''

function p = bvtl( nu, dh, dk, r )
%BVTL
%      p = bvtl( nu, dh, dk, r )
%    A function for computing bivariate t probabilities.
%    bvtl calculates the probability that x < dh and y < dk;
%   Parameters
%     nu integer number of degrees of freedom, nu < 1, gives Normal case
%     dh 1st upper integration limit
%     dk 2nd upper integration limit
%     r   correlation coefficient
%   Example: p = bvtl( 6, 3, 4, .35 )  

%
%        This function is based on the method described by 
%          Dunnett, C.W. and M. Sobel, (1954),
%          A bivariate generalization of Student's t-distribution
%          with tables for certain special cases,
%          Biometrika 41, pp. 153-169,
%
%       Alan Genz
%       Department of Mathematics
%       Washington State University
%       Pullman, Wa 99164-3113
%       Email : alangenz@wsu.edu
%
  if nu < 1, p = bvnl( dh, dk, r );
  elseif dh == -inf | dk == -inf, p = 0;
  elseif dh == inf, if dk == inf, p = 1; else p = studnt( nu, dk ); end
  elseif dk == inf, p = studnt( nu, dh );
  elseif 1 - r < eps, p = studnt( nu, min([ dh dk ]) );
  elseif r + 1 < eps, p = 0;  
    if dh > -dk, p = studnt( nu, dh ) - studnt( nu, -dk ); end 
  else, tpi = 2*pi; ors = 1 - r*r; hrk = dh - r*dk; krh = dk - r*dh; 
    if abs(hrk) + ors > 0 
      xnhk = hrk^2/( hrk^2 + ors*( nu + dk^2 ) );
      xnkh = krh^2/( krh^2 + ors*( nu + dh^2 ) );
    else, xnhk = 0; xnkh = 0; 
    end, hs = sign( dh - r*dk ); ks = sign( dk - r*dh );
    if mod( nu, 2 ) == 0
      bvt = atan2( sqrt(ors), -r )/tpi;
      gmph = dh/sqrt( 16*( nu + dh^2 ) ); gmpk = dk/sqrt( 16*( nu + dk^2 ) ); 
      btnckh = 2*atan2( sqrt( xnkh ), sqrt( 1 - xnkh ) )/pi; 
      btpdkh = 2*sqrt( xnkh*( 1 - xnkh ) )/pi;
      btnchk = 2*atan2( sqrt( xnhk ), sqrt( 1 - xnhk ) )/pi; 
      btpdhk = 2*sqrt( xnhk*( 1 - xnhk ) )/pi;
      for j = 1 : nu/2
    bvt = bvt + gmph*( 1 + ks*btnckh ); 
    bvt = bvt + gmpk*( 1 + hs*btnchk );
    btnckh = btnckh + btpdkh; btpdkh = 2*j*btpdkh*( 1 - xnkh )/(2*j+1); 
    btnchk = btnchk + btpdhk; btpdhk = 2*j*btpdhk*( 1 - xnhk )/(2*j+1); 
    gmph = gmph*( j - 1/2 )/( j*( 1 + dh^2/nu ) );
    gmpk = gmpk*( j - 1/2 )/( j*( 1 + dk^2/nu ) );
      end
    else, qhrk = sqrt( dh^2 + dk^2 - 2*r*dh*dk + nu*ors ); 
      hkrn = dh*dk + r*nu; hkn = dh*dk - nu; hpk = dh + dk;
      bvt = atan2( -sqrt(nu)*(hkn*qhrk+hpk*hkrn), hkn*hkrn-nu*hpk*qhrk )/tpi; 
      if bvt < -10*eps, bvt = bvt + 1; end
      gmph = dh/( tpi*sqrt(nu)*( 1 + dh^2/nu ) ); 
      gmpk = dk/( tpi*sqrt(nu)*( 1 + dk^2/nu ) ); 
      btnckh = sqrt( xnkh ); btpdkh = btnckh;
      btnchk = sqrt( xnhk ); btpdhk = btnchk; 
      for j = 1 : ( nu - 1 )/2
    bvt = bvt + gmph*( 1 + ks*btnckh ); 
    bvt = bvt + gmpk*( 1 + hs*btnchk );
    btpdkh = (2*j-1)*btpdkh*( 1 - xnkh )/(2*j); btnckh = btnckh + btpdkh; 
    btpdhk = (2*j-1)*btpdhk*( 1 - xnhk )/(2*j); btnchk = btnchk + btpdhk; 
    gmph = gmph*j/( ( j + 1/2 )*( 1 + dh^2/nu ) );
    gmpk = gmpk*j/( ( j + 1/2 )*( 1 + dk^2/nu ) );
      end
    end, p = bvt;
  end
%
%  end bvtl
%
function st = studnt( nu, t )
%
%     Student t Distribution Function;
%
%                       t
%         studnt = c   i  ( 1 + y*y/nu )^( -(nu+1)/2 ) dy
%                   nu -inf
%  
  if t == inf, st = 1; 
  elseif t == -inf, st = 0; 
  elseif nu  < 1, st = phid(t);
  elseif nu == 1, st = ( 1 + 2*atan(t)/pi )/2;
  elseif nu == 2, st = ( 1 + t/sqrt( 2 + t*t ))/2;
  else, tt = t*t; cssthe = 1/( 1 + tt/nu ); polyn = 1;
    for j = nu-2 : -2 : 2, polyn = 1 + ( j - 1 )*cssthe*polyn/j; end 
    if mod( nu, 2 ) == 1, rn = nu; ts = t/sqrt(rn);
      st = ( 1 + 2*( atan(ts) + ts*cssthe*polyn )/pi )/2;
    else, snthe = t/sqrt( nu + tt ); st = ( 1 + snthe*polyn )/2;
    end
    st = max( [ 0 min( [st 1] )] );
  end
%
% end studnt
%
function p = bvnl( dh, dk, r )
%BVNL
  p = bvnu( -dh, -dk, r );
%
%   end bvnl
%
function p = bvnu( dh, dk, r )
%BVNU
%  A function for computing bivariate normal probabilities.
%  bvnu calculates the probability that x > dh and y > dk. 
%    parameters  
%      dh 1st lower integration limit
%      dk 2nd lower integration limit
%      r   correlation coefficient
%  Example: p = bvnu( -3, -1, .35 )
%  Note: to compute the probability that x < dh and y < dk, 
%        use bvnu( -dh, -dk, r ). 
%
%   Author
%       Alan Genz
%       Department of Mathematics
%       Washington State University
%       Pullman, Wa 99164-3113
%       Email : alangenz@wsu.edu
%
%    This function is based on the method described by 
%        Drezner, Z and G.O. Wesolowsky, (1989),
%        On the computation of the bivariate normal inegral,
%        Journal of Statist. Comput. Simul. 35, pp. 101-107,
%    with major modifications for double precision, for |r| close to 1,
%    and for Matlab by Alan Genz. Minor bug modifications 7/98, 2/10.
%
  if dh == inf | dk == inf, p = 0;
  elseif dh == -inf, if dk == -inf, p = 1; else p = phid(dk); end
  elseif dk == -inf, p = phid(dh);
  else
    if abs(r) < 0.3, ng = 1; lg = 3;
      %       Gauss Legendre points and weights, n =  6
      w(1:3,1) = [0.1713244923791705 0.3607615730481384 0.4679139345726904]';
      x(1:3,1) = [0.9324695142031522 0.6612093864662647 0.2386191860831970]';
    elseif abs(r) < 0.75,  ng = 2; lg = 6;
      %       Gauss Legendre points and weights, n = 12
      w(1:3,2) = [.04717533638651177 0.1069393259953183 0.1600783285433464]';
      w(4:6,2) = [0.2031674267230659 0.2334925365383547 0.2491470458134029]';
      x(1:3,2) = [0.9815606342467191 0.9041172563704750 0.7699026741943050]';
      x(4:6,2) = [0.5873179542866171 0.3678314989981802 0.1252334085114692]';
    else, ng = 3; lg = 10;
      %       Gauss Legendre points and weights, n = 20
      w(1:3,3) = [.01761400713915212 .04060142980038694 .06267204833410906]';
      w(4:6,3) = [.08327674157670475 0.1019301198172404 0.1181945319615184]';
      w(7:9,3) = [0.1316886384491766 0.1420961093183821 0.1491729864726037]';
      w(10,3) = 0.1527533871307259;
      x(1:3,3) = [0.9931285991850949 0.9639719272779138 0.9122344282513259]';
      x(4:6,3) = [0.8391169718222188 0.7463319064601508 0.6360536807265150]';
      x(7:9,3) = [0.5108670019508271 0.3737060887154196 0.2277858511416451]';
      x(10,3) = 0.07652652113349733;
    end
    h = dh; k = dk; hk = h*k; bvn = 0;
    if abs(r) < 0.925, hs = ( h*h + k*k )/2; asr = asin(r);  
      for i = 1 : lg
    sn = sin( asr*( 1 - x(i,ng) )/2 );
    bvn = bvn + w(i,ng)*exp( ( sn*hk - hs )/( 1 - sn*sn ) );
    sn = sin( asr*( 1 + x(i,ng) )/2 );
    bvn = bvn + w(i,ng)*exp( ( sn*hk - hs )/( 1 - sn*sn ) );
      end, bvn = bvn*asr/( 4*pi );  
      bvn = bvn + phid(-h)*phid(-k);  
    else, twopi = 2*pi; if r < 0, k = -k; hk = -hk; end
      if abs(r) < 1
    as = ( 1 - r )*( 1 + r ); a = sqrt(as); bs = ( h - k )^2;
    c = ( 4 - hk )/8 ; d = ( 12 - hk )/16; asr = -( bs/as + hk )/2;
    if asr > -100  
      bvn = a*exp(asr)*( 1 - c*(bs-as)*(1-d*bs/5)/3 + c*d*as*as/5 );
    end
    if hk > -100, b = sqrt(bs); sp = sqrt(twopi)*phid(-b/a);
      bvn = bvn - exp(-hk/2)*sp*b*( 1 - c*bs*( 1 - d*bs/5 )/3 );
    end, a = a/2;
    for i = 1 : lg
      for is = -1 : 2 : 1, xs = ( a + a*is*x(i,ng) )^2; 
        rs = sqrt( 1 - xs ); asr = -( bs/xs + hk )/2;
        if asr > -100, sp = ( 1 + c*xs*( 1 + d*xs ) );
          ep = exp( -hk*( 1 - rs )/( 2*( 1 + rs ) ) )/rs;
          bvn = bvn + a*w(i,ng)*exp(asr)*( ep - sp );
        end
      end
    end, bvn = -bvn/twopi;
      end
      if r > 0, bvn =  bvn + phid( -max( h, k ) ); end
      if r < 0, bvn = -bvn + max( 0, phid(-h)-phid(-k) ); end
    end, p = max( 0, min( 1, bvn ) );
  end
%
%   end bvnu
%
function p = phid(z), p = erfc( -z/sqrt(2) )/2; % Normal cdf
%
% end phid
%
     
'''
