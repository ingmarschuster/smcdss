'''
    
    @author Christian Schäfer
'''

# $Date$
__version__ = "$Revision$"

from time import clock
from binary import productBinary
from auxpy.data import *
from numpy import *
from scipy.linalg import cholesky, eigvalsh, eigh
from scipy.stats import norm, rv_continuous


CONST_PRECISION = 0.00001
CONST_ITERATIONS = 30


class hiddenNormalBinary(productBinary):
    '''
        A multivariate Bernoulli as function of a hidden multivariate normal distribution.
    '''

    def __init__(self, p, R):
        '''
            Constructor.
            @param p mean
            @param R correlation matrix
        '''
        productBinary.__init__(self, p, name='hidden-normal-binary', longname='A multivariate Bernoulli as function of a hidden multivariate normal distribution.')

        ## correlation matrix of the binary distribution
        self.R = R
        ## mean of hidden normal distribution
        self.mu = norm.ppf(self.p)

        localQ = calcLocalQ(R, self.mu, self.p)
        ## correlation matrix of the hidden normal distribution
        self.C, self.Q = decomposeQ(localQ, mode='scaled', verbose=False)

    @classmethod
    def random(cls, d):
        '''
            Constructs a random hidden-normal-binary model for testing.
            @param cls class 
            @param d dimension
        '''
        p = 0.3 + 0.4 * random.rand(d)

        # For a random matrix X with entries U[-1,1], set Q = X*X^t and normalize.
        X = ones((d, d)) - 2 * random.random((d, d))
        Q = dot(X, X.T) + exp(-10) * eye(d)
        q = Q.diagonal()[newaxis, :]
        Q = Q / sqrt(dot(q.T, q))
        R = calcR(Q, norm.ppf(p), p)

        return cls(p, R)

    @classmethod
    def independent(cls, p):
        '''
            Constructs a hidden-normal-binary model with independent components.
            @param cls class 
            @param p mean
        '''
        return cls(p, eye(len(p)))

    @classmethod
    def fromData(cls, sample):
        '''
            Construct a product-binary model from data.
            @param cls class
            @param sample a sample of binary data
        '''
        return cls(sample.mean, sample.cor)


    def __pmf(self, gamma):
        '''
            Probability mass function. Not available.
            @param gamma binary vector
        '''
        raise ValueError("No evaluation of the pmf for the normal-binary model.")

    def __lpmf(self, gamma):
        '''
            Log-probability mass function. Not available.
            @param gamma binary vector    
        '''
        raise ValueError("No evaluation of the pmf for the normal-binary model.")

    def __rvs(self):
        '''
            Samples from the model.
            @return random variate
        '''
        if self.d == 0: return
        v = random.normal(size=self.d)
        return dot(self.C, v) < self.mu

    def __rvslpmf(self):
        '''
            Samples from the model and evaluates the likelihood of the sample. Not available.
            @return random variate
            @return likelihood
        '''
        rv = self.rvs()
        return rv, 0

    def __str__(self):
        return format_vector(self.p, 'p') + '\n' + format_matrix(self.R, 'R')




def calcR(Q, mu, p):
    '''
        Computes the hidden-normal-binary correlation matrix R induced by
        the hidden-normal correlation matrix Q.
        @param Q correlation matrix of the hidden normal
        @param mu mean of the hidden normal
        @param p mean of the binary
    '''
    d = len(p)
    R = ones((d, d))
    for i in range(d):
        for j in range(i):
            R[i][j] = bvnorm.pdf([mu[i], mu[j]], Q[i][j]) - p[i] * p[j]
            R[i][j] /= sqrt(p[i] * p[j] * (1 - p[i]) * (1 - p[j]))
            R[i][j] = max(min(R[i][j], 1.0), -1.0)
        R[0:i, i] = R[i, 0:i].T
    return R


def calcLocalQ(R, mu, p, verbose=False):
    '''
        Computes the hidden-normal correlation matrix Q necessary to generate
        bivariate bernoulli samples with a certain local correlation matrix R.
        @param R correlation matrix of the binary
        @param mu mean of the hidden normal
        @param p mean of the binary
        @param verbose print to stdout 
    '''
    t = clock()
    iter = 0
    d = len(p)
    localQ = ones((d, d))

    for i in range(d):
        for j in range(i):
            localQ[i][j], n = \
                calcLocalq(mu=[mu[i], mu[j]], p=[p[i], p[j]], r=R[i][j], init=R[i][j])
            iter += n
        localQ[0:i, i] = localQ[i, 0:i].T

    if verbose: print 'calcLocalQ'.ljust(20) + '> time %.3f, loops %i' % (clock() - t, iter)

    return localQ


def calcLocalq(mu, p, r, init=0, verbose=False):
    '''
        Computes the hidden-normal correlation q necessary to generate
        bivariate bernoulli samples with a certain correlation r.
        @param mu mean of the hidden normal
        @param p mean of the binary            
        @param r correlation between the binary
        @param init initial value
        @param verbose print to stdout 
    '''

    # For extreme marginals, correlation is negligible.
    for i in range(2):
        if abs(mu[i]) > 3.2: return 0.0, 0

    # Ensure r is feasible.
    t = sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
    maxr = min(0.999, (min(p[0], p[1]) - p[0] * p[1]) / t)
    minr = max(-0.999, (max(p[0] + p[1] - 1, 0) - p[0] * p[1]) / t)
    r = min(max(r, minr), maxr)

    # Solve implicit form by iteration.
    q, iter = newtonraphson(mu, p, r, init)
    if q < -1:
        q, n = bisectional(mu, p, r, l= -1, u=0, init= -0.5)
        iter += n

    if q == inf or isnan(q): q = 0
    q = max(min(q, 0.999), -0.999)

    return q, iter


def newtonraphson(mu, p, r, init=0, verbose=False):
    '''
        Newton-Raphson search for the correlation parameter q of the underlying normal distibution.
        @param mu mean of the hidden normal
        @param p mean of the binary            
        @param r correlation between the binary
        @param init initial value
        @param verbose print to stdout 
    '''
    if verbose: print '\nNewton-Raphson search.'

    t = sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
    s = p[0] * p[1] + r * t

    greater_one = False
    q = init
    last_q = inf

    for iter in range(CONST_ITERATIONS):
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

        if abs(last_q - q) < CONST_PRECISION: break
        last_q = q

    return q, iter


def bisectional(mu, p, r, l= -1, u=1, init=0, verbose=False):
    '''
        Bisectional search for the correlation parameter q of the underlying normal distibution.
        @param mu mean of the hidden normal
        @param p mean of the binary            
        @param r correlation between the binary
        @param l lower bound
        @param u upper bound         
        @param init initial value
        @param verbose print to stdout 
    '''
    if verbose: print '\nBisectional search.'

    t = sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
    q = init

    for iter in range(CONST_ITERATIONS):
        if verbose: print q
        v = (bvnorm.cdf(mu, q) - p[0] * p[1]) / t
        if r < v:
            u = q; q = 0.5 * (q + l)
        else:
            l = q; q = 0.5 * (q + u)
        if abs(l - u) < CONST_PRECISION: break

    return q, iter


def decomposeQ(Q, mode='scaled', verbose=False):
    '''
        Computes the Cholesky decompostion of Q. If Q is not positive definite, either the
        identity matrix, a scaled version of Q or the correlation matrix nearest to Q is used.
        @param Q summetric matrix 
        @param mode way of dealing with non-definite matrices [independent, scaled, nearest] 
        @param verbose print to stdout 
    '''
    t = clock()
    d = len(Q[0])
    try:
        C = cholesky(Q, True)
    except:
        if mode == 'independent':
            return eye(d), eye(d)
        if mode == 'scaled':
            Q = scaleQ(Q, verbose=verbose)
        if mode == 'nearest':
            Q = nearestQ(Q, verbose=verbose)

    try:
        C = cholesky(Q, True)
    except:
        print "WARNING: Set matrix to identity."
        C, Q = eye(d), eye(d)

    if verbose: print 'decomposeQ'.ljust(20) + '> time %.3f' % (clock() - t)

    return C, Q


def scaleQ(Q, verbose=False):
    '''
        Rescales the locally adjusted matrix Q to make it positive definite.
        @param Q summetric matrix 
        @param verbose print to stdout 
    '''
    t = clock()
    d = len(Q[0])
    try:
        n = abs(Q).sum() - d
    except:
        print "WARNING: Could not evaluate norm."
        n = 1.0

    # If the smallest eigenvalue is (almost) negative, rescale Q matrix.
    mineig = min(eigvalsh(Q)) - exp(-5.0)
    if mineig < 0:
        Q -= mineig * eye(d)
        Q /= (1 - mineig)

    if verbose: print 'scaleQ'.ljust(20) + '> time %.3f, ratio %.3f' % (clock() - t, (abs(Q).sum() - d) / n)

    return Q


def nearestQ(Q, verbose=False):
    '''
        Computes the nearest (Frobenius norm) correlation matrix for the locally adjusted matrix Q.
        The nearest correlation matrix problem is solved using the alternating projection method proposed
        in <i>Computing the Nearest Correlation Matrix - A problem from Finance</i> by N. Higham (2001).
        @param Q summetric matrix 
        @param verbose print to stdout 
    '''
    t = clock()
    d = len(Q[0])
    try:
        n = abs(Q).sum() - d
    except:
        print "WARNING: Could not evaluate norm."
        n = 1.0

    # run alternating projections
    S = zeros((d, d))
    Y = Q
    for iter in range(CONST_ITERATIONS):
        # Dykstra's correction term
        R = Y - S

        # Project corrected Y matrix on convex set of positive definite matrices.
        D, E = eigh(R)
        for j in range(len(D)):
            D[j] = max(D[j], exp(-8.0))
        X = dot(dot(E, diag(D)), E.T)

        # Update correction term.
        S = X - R

        # Project X matrix on convex set of matrices with unit diagonal.
        Y = X.copy()
        for j in range(len(D)): Y[j][j] = 1.

        if linalg.norm(X - Y) < 0.001: break

    q = diagonal(X)[newaxis, :]
    Q = X / sqrt(dot(q.T, q))
    if verbose: print 'nearestQ'.ljust(20) + '> time %.3f, loops %i, ratio %.3f' % (clock() - t, iter, (abs(Q).sum() - d) / n)

    return Q





class _bvnorm(rv_continuous):
    '''
        Bivariate normal distribution with correlation r.
        normal.pdf(x,y) = exp(-(x*x-2*r*x*y+y*y)/(2*(1-r*r))) / (2*pi*sqrt(1-r*r))
    '''

    def cdf(self, x, r=0):
        '''
            Computes the bivariate normal cumulative distribution function,
            i.e. the probability that X < x and Y < y. The function only calls lowerDW(x, y, r).
            @param x value
            @param r correlation coefficient 
        '''
        return self.lowerDW(x[0], x[1], r)

    def pdf(self, x, r=0):
        '''
            Computes the bivariate normal probability distribution function, i.e. the density at (x, y)
            @param x value
            @param r correlation coefficient 
        '''
        z = x[0] * x[0] - 2 * r * x[0] * x[1] + x[1] * x[1]
        return exp(-z / (2 * (1 - r * r))) / (2 * pi * sqrt(1 - r * r))

    def rvs(self, r=0):
        '''
            @param r correlation coefficient 
            @return random bivariate normal
        '''
        v = random.normal(0, 1)
        return r * v + sqrt(1 - r * r) * random.normal(0, 1)

    def lowerDW(self, dh, dk, r):
        '''
            Computes bivariate normal probabilities; lowerDW calculates the probability
            that x < dh and y < dk using the Drezner-Wesolowsky approximation.
            The function only calls upperDW(-dh, -dk, r).
            
            @param dh 1st lower integration limit
            @param dk 2nd lower integration limit
            @param r correlation coefficient
        '''
        return self.upperDW(-dh, -dk, r)

    def upperDW(self, dh, dk, r):
        '''
            Computes bivariate normal probabilities; upperDW calculates the probability that x > dh and y > dk. 
              
            This function is based on the method described by Z. Drezner and G.O. Wesolowsky, (1989),
            "On the computation of the bivariate normal integral", Journal of Statist.
            Comput. Simul. 35, pp. 101-107, with major modifications for double precision, for |r| close to 1.
        
            The code was adapted for python from the matlab routine by Alan Genz.
            
            @param dh 1st lower integration limit
            @param dk 2nd lower integration limit
            @param r correlation coefficient
        '''
        twopi = 2 * pi
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
            hs = (h * h + k * k) / 2; asr = arcsin(r);
            for i in range(lg):
                sn = sin(asr * (1 - x[i]) / 2);
                bvn = bvn + w[i] * exp((sn * hk - hs) / (1 - sn * sn));
                sn = sin(asr * (1 + x[i]) / 2);
                bvn = bvn + w[i] * exp((sn * hk - hs) / (1 - sn * sn));
            bvn = bvn * asr / (4 * pi) + norm.cdf(-h) * norm.cdf(-k)
        else:
            if r < 0:
                    k = -k
                    hk = -hk
            if abs(r) < 1:
                aas = (1 - r) * (1 + r); a = sqrt(aas); bs = (h - k) ** 2;
                c = (4 - hk) / 8 ; d = (12 - hk) / 16; asr = -(bs / aas + hk) / 2;
                if asr > -100:
                    bvn = a * exp(asr) * (1 - c * (bs - aas) * (1 - d * bs / 5) / 3 + c * d * aas * aas / 5);
                if - hk < 100:
                    b = sqrt(bs); sp = sqrt(twopi) * norm.cdf(-b / a);
                    bvn = bvn - exp(-hk / 2) * sp * b * (1 - c * bs * (1 - d * bs / 5) / 3);
                a = a / 2;
                for i in range(lg):
                    for iis in range(-1 , 3 , 2):
                        xs = (a * (iis * x[i] + 1)) ** 2; rs = sqrt(1 - xs);
                        asr = -(bs / xs + hk) / 2;
                        if asr > -100:
                            sp = (1 + c * xs * (1 + d * xs));
                            ep = exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs;
                            bvn = bvn + a * w[i] * exp(asr) * (ep - sp);
                bvn = -bvn / twopi;
            if r > 0: bvn = bvn + norm.cdf(-max(h, k))
            if r < 0: bvn = -bvn + max(0, norm.cdf(-h) - norm.cdf(-k))
        p = max(0, min(1, bvn));
        return p

    def lowerMC(self, dh, dk, r, n=100000):
        '''
            Computes bivariate normal probabilities; lowerMC calculates the probability that x < dh and y < dk
            using a Monte Carlo approximation of n samples. The function only calls upperMC(-dh, -dk, r, n).

            @param dh 1st lower integration limit
            @param dk 2nd lower integration limit
            @param r   correlation coefficient
            @param n   sample size
        '''
        return self.upperMC(-dh, -dk, r, n)

    def upperMC(self, dh, dk, r, n=100000):
        '''
            Computes bivariate normal probabilities; upperMC calculates the probability that x > dh and y > dk. 
            This function is a simple MC evaluation used to cross-check the DW approximation algorithms.
        
            @param dh 1st lower integration limit
            @param dk 2nd lower integration limit
            @param r   correlation coefficient
            @param n   sample size
        
        '''
        p = 0
        for i in range(n):
            v1 = random.normal(0, 1)
            v2 = r * v1 + sqrt(1 - r * r) * random.normal(0, 1)
            if v1 > dh and v2 > dk:p += 1
        return p / float(n)

bvnorm = _bvnorm(name='bvnorm', longname='A bivariate normal', shapes='r')
