'''
@author: cschafer
'''

from time import clock
from binary import product_binary
from auxpy.format import *
from bvnorm import bvnorm
from numpy import *
from scipy.linalg import cholesky, eigvalsh, eigh, solve
from scipy.stats import norm

constPrecision = 10 ** -5
constIteration = 30

class hnormal_binary(product_binary):
    ''' Binary model based on a hidden normal distribution. '''
    def __init__(self, p, R):
        product_binary.__init__(self, p)
        self.name = 'hidden-normal-binary'
        self.R = R

        self.mu = norm.ppf(self.p)
        localQ = self.calcLocalQ(R, self.mu, self.p)
        self.C, self.Q = self.decomposeQ(localQ, mode='scaled', verbose=False)

    def pmf(self, gamma):
        ''' Probability mass function. Not available. '''
        raise ValueError("No evaluation of the pmf for the normal-binary model.")

    def lpmf(self, gamma):
        ''' Log-probability mass function. Not available. '''
        raise ValueError("No evaluation of the pmf for the normal-binary model.")

    def rvs(self):
        ''' Generates a random variate. '''
        if self.d == 0: return
        v = random.normal(size=self.d)
        return dot(self.C, v) < self.mu
    
    def rvslpmf(self):
        ''' Generates a random variate and computes its likelihood. Not available.'''
        rv = self.rvs()
        return rv, None

    def autotest(self, n):
        ''' Generates a sample of size n. Compares the empirical and the true first moments. '''
        sample = []
        for i in range(n):
            sample.append(array(self.rvs(), dtype=float))
        sample = array(sample)

        mean = sample.sum(axis=0) / n
        print 'true p\n' + format_vector(self.p)
        print 'sample mean\n' + format_vector(mean)

        cov = (dot(sample.T, sample) - n * dot(mean[:, newaxis], mean[newaxis, :])) / (n - 1)
        var = cov.diagonal()[newaxis, :]
        cor = cov / sqrt(dot(var.T, var))
        print 'true R\n' + format_matrix(self.R)
        print 'cor R\n' + format_matrix(cor)
        print 'hidden Q\n' + format_matrix(self.Q)



    @classmethod
    def random(cls, d):
        ''' Construct a random hidden-normal-binary model for testing.'''
        p = 0.3 + 0.4 * random.rand(d)

        # For a random matrix X with entries U[-1,1], set Q = X*X^t and normalize.
        X = ones((d, d)) - 2 * random.random((d, d))
        Q = dot(X, X.T) + exp(-10) * eye(d)
        q = Q.diagonal()[newaxis, :]
        Q = Q / sqrt(dot(q.T, q))
        R = hnormal_binary.calcR(Q, norm.ppf(p), p)

        return cls(p, R)



    @staticmethod
    def calcR(Q, mu, p):
        '''
        Computes the hidden-normal-binary correlation matrix R induced by
        the hidden-normal correlation matrix Q.
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


    @staticmethod
    def calcLocalQ(R, mu, p, verbose=False):
        '''
        Computes the hidden-normal correlation matrix Q necessary to generate
        bivariate bernoulli samples with a certain local correlation matrix R.
        '''
        t = clock()
        iter = 0
        d = len(p)
        localQ = ones((d, d))

        for i in range(d):
            for j in range(i):
                localQ[i][j], n = \
                    hnormal_binary.calcLocalq(mu=[mu[i], mu[j]], r=R[i][j], p=[p[i], p[j]], init=R[i][j])
                iter += n
            localQ[0:i, i] = localQ[i, 0:i].T

        if verbose: print 'calcLocalQ'.ljust(20) + '> time %.3f, loops %i' % (clock() - t, iter)

        return localQ


    @staticmethod
    def calcLocalq(mu, r, p, init=0, verbose=False):
        '''
        Computes the hidden-normal correlation q necessary to generate
        bivariate bernoulli samples with a certain correlation r.            
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
        q, iter = hnormal_binary.newtonraphson(mu, p, r, init)
        if q < -1:
            q, n = hnormal_binary.bisectional(mu, p, r, l= -1, u=0, init= -0.5)
            iter += n

        if q == inf or isnan(q): q = 0
        q = max(min(q, 0.999), -0.999)

        return q, iter


    @staticmethod
    def newtonraphson(mu, p, r, init=0, verbose=False):
        '''
        Newton-Raphson search for the correlation parameter q of the underlying normal distibution.
        '''
        if verbose: print '\nNewton-Raphson search.'

        t = sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
        s = p[0] * p[1] + r * t

        greater_one = False
        q = init
        last_q = inf

        for iter in range(constIteration):
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

            if abs(last_q - q) < constPrecision: break
            last_q = q

        return q, iter


    @staticmethod
    def bisectional(mu, p, r, l= -1, u=1, init=0, verbose=False):
        '''
        Bisectional search for the correlation parameter q of the underlying normal distibution.
        '''
        if verbose: print '\nBisectional search.'

        t = sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
        q = init

        for iter in range(constIteration):
            if verbose: print q
            v = (bvnorm.cdf(mu, q) - p[0] * p[1]) / t
            if r < v:
                u = q; q = 0.5 * (q + l)
            else:
                l = q; q = 0.5 * (q + u)
            if abs(l - u) < constPrecision: break

        return q, iter


    @staticmethod
    def decomposeQ(Q, mode='scaled', verbose=False):
        '''
        Computes the Cholesky decompostion of Q. If Q is not positive definite, either the
        identity matrix, a scaled version of Q or the correlation matrix nearest to Q is used.
        '''
        t = clock()
        d = len(Q[0])
        try:
            C = cholesky(Q, True)
        except:
            if mode == 'independent':
                return eye(d), eye(d)
            if mode == 'scaled':
                Q = hnormal_binary.scaleQ(Q, verbose=verbose)
            if mode == 'nearest':
                Q = hnormal_binary.nearestQ(Q, verbose=verbose)

        try:
            C = cholesky(Q, True)
        except:
            print "WARNING: Set matrix to identity."
            C, Q = eye(d), eye(d)

        if verbose: print 'decomposeQ'.ljust(20) + '> time %.3f' % (clock() - t)

        return C, Q


    @staticmethod
    def scaleQ(Q, verbose=False):
        '''
        Rescales the locally adjusted matrix Q to make it positive definite.
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


    @staticmethod
    def nearestQ(Q, max_iter=30, verbose=False):
        '''
        Computes the nearest (Frobenius norm) correlation matrix for the locally adjusted matrix Q.
        
        The nearest correlation matrix problem is solved using the alternating projection method proposed
        in 'Computing the Nearest Correlation Matrix - A problem from Finance' by N. Higham (2001)
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
        for iter in range(max_iter):
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
