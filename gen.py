'''
Created on 30 nov. 2009

@author: cschafer
'''

from numpy.random import rand, normal as normal_rand
from numpy import *
from numpy.linalg import norm
from scipy.linalg import cholesky, eigvalsh, solve
from bvnorm import bvnorm as bvnormal
from scipy.stats import rv_discrete, norm as normal
from scipy.stats.distributions import inf
from copy import *
import csv, types, time, platform

if platform.system() == 'Linux':
    import scipy.weave as weave
    WEAVE = True
else:
    WEAVE = False

from sampling import *

## precision for breakin
DECIMALS = 10 ** -4
## maximum number of loops
LOOPS = 25
## maximum size feasible for exhaustive exploration
MAX_FEASIBLE = 12
PATH = "/home/cschafer/Eden/Documents/Python/workspace"

try:
    from rpy import *
    hasrpy = True
except:
    hasrpy = False

def bin2dec(b):
    '''
    Converts a boolean array into an integer.
    '''
    return long(bin2str(b), 2)

def bin2str(b):
    '''
    Converts a boolean array into a binary string.
    '''
    return ''.join([str(i) for i in array(b, dtype=int)])

def dec2bin(n, dim=0):
    '''
    Converts an integer into a boolean array containing its binary representation.
    '''
    b = []
    while n > 0:
        if n % 2:
            b.append(True)
        else:
            b.append(False)            
        n = n >> 1
    while len(b) < dim:
        b.append(False)
    b.reverse()
    return array(b)

def crosscols(firstcol, lastcol):
    '''
    Returns an array of all pairs between firstcol and lastcol.
    '''
    p = []
    for i in range(lastcol - firstcol + 1):
        p.append(firstcol + i)   
        for j in range(firstcol + i, lastcol + 1):
            p.append([firstcol + i, j])
    return p

class binary(rv_discrete):
    '''
    The generator interface to be implemented by generators.
    '''
    def __init__(self, data=None, fraction_mean=1, fraction_corr=1, smooth_mean=0, smooth_corr=0, \
                 threshold_randomness=0, p=None, mean=None, min_p=0, weighted=False, verbose=False):
        
        rv_discrete.__init__(self, name='binary')
        self.nozero = True
        self.dim = None
        self.min_p = min_p
        self.adjusted = []
        self.data = data
        self.fraction_mean = fraction_mean; self.fraction_corr = fraction_corr
        self.smooth_mean = smooth_mean; self.smooth_corr = smooth_corr
        self.threshold_randomness = threshold_randomness
        self.weighted = weighted; self.verbose = verbose
        self.p = p
     
        if not self.data == None: mean = self.data.mean(fraction=self.fraction_mean, weighted=weighted)
        if mean == None: return
                
        # smooth the full mean
        if hasattr(self, "mean"): previous_mean = self.mean
        else: previous_mean = .5 * ones(self.p)
        self.mean = self.smooth_mean * previous_mean + (1 - self.smooth_mean) * mean
              
        # remember last iterations strongly random variables
        if hasattr(self, "strongly_random"): self.previous_strongly_random = list(self.strongly_random)
        else: self.previous_strongly_random = range(len(self.mean))
        if hasattr(self, "weakly_random"): self.previous_weakly_random_random = list(self.weakly_random)
        else: self.previous_weakly_random = []

        # separate weakly and strongly random variables
        self.dim = len(self.mean)
        self.weakly_random = []; self.strongly_random = [];
        for i in range(self.dim):
            if self.mean[i] < self.threshold_randomness or self.mean[i] > 1. + 1e-10 - self.threshold_randomness: self.weakly_random.append(i)
            else: self.strongly_random.append(i)
        
        # shrink dataset to strongly random variables
        self.p = len(self.strongly_random)
        if self.dim > self.p and not self.data == None: self.data.index = self.strongly_random
    
    def reset(self, data=None):
        for attr in ['p', 'data']:
            if hasattr(self, attr): delattr(self, attr)
        self.__init__(data=data, fraction_mean=self.fraction_mean, fraction_corr=self.fraction_corr, \
                      smooth_mean=self.smooth_mean, smooth_corr=self.smooth_corr, threshold_randomness=self.threshold_randomness, \
                      min_p=self.min_p, weighted=self.weighted, verbose=self.verbose)

    def pmf(self, gamma):
        return self._pmf(gamma)
    
    def lpmf(self, gamma):
        return log(self._pmf(gamma))

    def rvs(self):
        '''
        Generate a random variable.
        '''       
        rv = self._rvs()
        if self.nozero:
            while not rv.any(): rv = self._expand_rv(self._rvs())
        return rv
    
    def _expand_01(self, rv, where_1=None):
        if self.dim == None or len(rv) == self.dim: return rv
        gamma = zeros(self.dim, dtype=bool)
        gamma[self.strongly_random] = rv
        if where_1 == None: where_1 = list(set(where(self.mean > 0.5)[0]) & set(self.weakly_random))
        gamma[where_1] = True
        return gamma

    def _expand_rv(self, rv, partial_logprob=None, previous=False):
        '''
        Expands vector rv to full dimension by adding independent Bernoulli variables.
        '''
        if self.dim == None or len(rv) == self.dim:
            if partial_logprob == None: return rv
            else: return rv, partial_logprob
        gamma = empty(self.dim, dtype=bool)
        if previous: indices = self.previous_weakly_random
        else: indices = self.weakly_random
        gamma[self.strongly_random] = rv
        gamma[indices] = self.mean[indices] > rand(len(indices))
        if partial_logprob == None: return gamma
        return gamma, self._expand_logprob(gamma, partial_logprob)
    
    def _expand_logprob(self, gamma, partial_logprob):
        '''
        Expands partial log probability to full log probability.
        '''
        if not len(gamma) == self.dim: raise NameError('variable dimension incorrect.')
        prob = 1.
        for i in self.weakly_random:
            if gamma[i]: prob *= self.mean[i]
            else: prob *= (1 - self.mean[i])
        return log(prob) + partial_logprob
        

    def plot(self):
        '''
        Plots the probability distribution function of the generator (if possible).
        '''
        if not hasrpy: return
        x = []; y = zeros(2 ** self.p)
        
        for d in range(2 ** self.p):
            b = dec2bin(d, self.p)
            x.append(str(b))
            y[d] = self.prob(b)
    
        # normalize
        y = y / sum(y)

        # plot with R
        r.pdf(paper="a4r", file=self.name + "_plot.pdf", width=12, height=12, title="")
        r.barplot(y, names=x, cex_names=4. / self.p, las=3)
        r.title(main=self.type)


    def hist(self, n=10000):
        '''
        Plots a histogram of the empirical distribution obtained by sampling from the generator.
        '''
        if not hasrpy: return
        x = []; y = zeros(2 ** self.p)
                
        for i in range(n):
            b = self.rvs()
            y[bin2dec(b)] += 1.
        
        for d in range(2 ** self.p):
            b = dec2bin(d, self.p)
            x.append(str(b))
    
        # normalize
        y = y / sum(y)
        
        # plot with R
        r.pdf(paper="a4r", file=self.name + "_hist.pdf", width=12, height=12, title="")
        r.barplot(y, names=x, cex_names=4. / self.p, las=3, col="lightblue")
        r.title(main=self.type)


class binary_ind(binary):
    '''
    Generates samples with independent components.
    '''
    def __init__(self, data=None, fraction_mean=1, fraction_corr=None, smooth_mean=0, smooth_corr=None, \
                 threshold_randomness=0, mean=None, p=None, min_p=0, weighted=False, verbose=False):

        self.uniform = 0

        # set a given, random or uniform law
        if data == None:
            if not mean == None:
                if isinstance(mean, str):
                    if not p == None:
                        if mean == "random":
                            mean = rand(p)
                        if mean == "uniform":
                            mean = 0.5 * ones(p)
                            self.uniform = 0.5 ** p
                else:
                    p = len(mean)
        
        # call superconstructor
        binary.__init__(self, data=data, fraction_mean=fraction_mean, smooth_mean=smooth_mean, threshold_randomness=threshold_randomness, \
                        p=p, mean=mean, min_p=min_p, weighted=weighted, verbose=verbose)
        self.name = "binary_ind"

    def _pmf(self, gamma):
        if self.uniform > 0: return self.uniform
        prob = 1.
        for i, mprob in enumerate(self.mean):
            if gamma[i]: prob *= mprob
            else: prob *= (1 - mprob)
        return prob

    def _rvs(self):
        if self.p == 0: return []
        rv = self.mean[self.strongly_random] > rand(self.p)
        return self._expand_rv(rv)

    def rvsplus(self):
        rv = self._rvs()
        return self._expand_rv(rv, self.lpmf(rv))

class binary_mn(binary):
    def __init__(self, data=None, fraction_mean=1, smooth_mean=0, fraction_corr=1, smooth_corr=0, threshold_randomness=0, \
                 mu="random", Q="independent", mean=None, R=None, p=None, min_p=0, weighted=False, verbose=False):
        '''
        Generates samples from the multivariate normal model.
        
            data    data object
            mu      mean vector (normal)
            Q       correlation matrix (normal)
            mean      probability vector (binary)
            R       correlation matrix (binary)
            p       dimension
            
            a       delay parameter
            b       fraction for estimation of R
            c       decrease correlation by R = (c*I+R)/(1+c)
            
            mu = "uniform", "random"
            Q  = "scaled", "nearest", "gaussian", "independent", "random"
            
        '''
        
        # set mean vector mu
        if mean == None:
            if isinstance(mu, str) and not p == None:
                if mu == "random":
                    mu = normal_rand(size=p)
                elif mu == "uniform":
                    mu = zeros(p)
                mean = normal.cdf(mu)
        
        binary.__init__(self, data=data, fraction_mean=fraction_mean, fraction_corr=fraction_corr, \
                        smooth_mean=smooth_mean, smooth_corr=smooth_corr, p=p, mean=mean, min_p=min_p, \
                        threshold_randomness=threshold_randomness, weighted=weighted, verbose=verbose)        
        self.name = "binary_mn"

        if self.p < self.min_p: return

        if not hasattr(self, "oldC"):
            self.oldC = eye(self.p); self.oldQ = eye(self.p); self.oldmu = zeros(self.p)
        else:
            self.oldC = self.C; self.oldQ = self.Q; self.oldmu = self.mu
        
        if not self.data == None:
            self.R = self.data.cor(fraction=fraction_corr)
            self.R += self.smooth_corr * eye(self.p)
            self.R /= (1 + self.smooth_corr)
        else:
            self.mean = mean
            self.R = R
  
        self.mu = normal.ppf(self.mean[self.strongly_random])

        # set dimension p
        if not Q == "independent": self.p = len(Q[0])
        if not mean == None: self.p = len(mean)
        if not R == None: self.p = len(R[0])
        if not p == None: self.p = p
        if self.p == None: return

        # set correlation matrix Q
        if self.R == None:
            if not isinstance(Q, str):
                # use explicitely given Q
                self.Q = Q
            elif Q == "random":
                # generate a random Q 
                self.randomQ()
            elif Q == "independent":
                # use the identity for Q
                self.independentQ()
        else:
            if Q == "nearest":
                # solve nearest correlation matrix problem for local Q matrix
                self.nearestQ()                
            else: # Q == "scaled"
                # scale local Q matrix until all eigenvalues are positive
                self.scaledQ()
        try:
            self.C = cholesky(self.Q, True)
        except:
            eigvals = eigvalsh(self.Q)
            print "WARNING: Eigenvalue at %.3f after scaling; set to identity." % min(eigvals)
            self.C = eye(self.p)

    def randomQ(self):
        '''
        Generate a random matrix X with entries U[-1,1]. Set Q = X*X^t and normalize.  
        '''
        X = ones((self.p, self.p)) - 2 * random.random((self.p, self.p))
        self.Q = dot(X, X.T) + 10 ** -5 * eye(self.p)
        q = self.Q.diagonal()[newaxis, :]
        self.Q = self.Q / sqrt(dot(q.T, q))
        self.R = self.Q2R()
    
    def independentQ(self):
        '''
        Set Q = I.
        '''
        self.Q, self.R = eye(self.p), eye(self.p)
         
    def scaledQ(self):
        '''
        Rescale the locally adjusted matrix localQ to make it positive definite.
        '''
        localQ = self.R2localQ()
        try:
            nlocalQ = (norm(localQ) ** 2 / self.p)
        except:
            print "WARNING: Could not evaluate norm."
            nlocalQ = 1.

        # get smallest eigenvalue from symmetric matrix
        eigvals = eigvalsh(localQ)
        delta = min(eigvals)
        # if min(eigvals) is (almost) negative, rescale localQ matrix
        self.Q = localQ
        if delta < 10 ** -4:
            delta = abs(delta) + 10 ** -4
            self.Q += delta * eye(self.p)
            self.Q /= (1 + delta)

        nQ = norm(self.Q) ** 2 / self.p
        self.adjusted.append("orig norm %.3f" % nlocalQ)        
        self.adjusted.append("final norm %.3f" % nQ)
        self.adjusted.append("ratio %.3f" % (nQ / nlocalQ))
        
    def nearestQ(self):
        '''
        Compute the nearest correlation matrix (Frobenius norm) for the locally adjusted matrix localQ.
        
        The nearest correlation matrix problem is solved using the alternating projection method proposed in "Computing the Nearest Correlation Matrix - A problem from Finance" by N. Higham (2001)
        '''
        localQ = self.R2localQ()
        nlocalQ = (norm(localQ) ** 2 / self.p)

        # run alternating projections
        S = zeros((self.p, self.p)) 
        Y = localQ
        for i in range(40):
            # Dykstra's correction term
            R = Y - S
            
            # project corrected Y matrix on convex set of positive definite matrices
            d, E = eigh(R)
            for j in range(len(d)): d[j] = max(d[j], 10 ** -5)
            X = dot(dot(E, diag(d)), E.T)
            
            # update correction term
            S = X - R
            
            # project X matrix on convex set of matrices with unit diagonal
            Y = X.copy()
            for j in range(len(d)): Y[j][j] = 1.
            
            if norm(X - Y) < 0.001: break
            
        q = diagonal(X)[newaxis, :]
        self.Q = X / sqrt(dot(q.T, q))
        
        nQ = (norm(self.Q) ** 2 / self.p)
        self.adjusted.append("orig norm %.3f" % nlocalQ)        
        self.adjusted.append("final norm %.3f" % nQ)
        self.adjusted.append("ratio %.3f" % (nQ / nlocalQ))
    
    def Q2R(self):
        '''
        Compute the binary correlation matrix R from the normal correlation matrix Q. 
        '''
        R = ones((self.p, self.p))
        for i in range(self.p):
            for j in range(i):
                R[i][j] = bvnormal.pdf([self.mu[i], self.mu[j]], self.Q[i][j])
                R[i][j] -= self.mean[i] * self.mean[j]
                R[i][j] /= sqrt(self.mean[i] * self.mean[j] * (1 - self.mean[i]) * (1 - self.mean[j]))
                R[i][j] = max(min(R[i][j], 1), -1)
            R[0:i, i] = R[i, 0:i].T
        return R
    
    def R2Q(self):
        '''
        Compute the locally adjusted matrix localQ from the binary correlation matrix R.
        '''
        self.R2localQ()
    
    def R2localQ(self):
        '''
        Compute the locally adjusted matrix localQ from the binary correlation matrix R.
        '''
        t = time.clock()
        localQ = ones((self.p, self.p))
        total_iter = 0
        for i in range(self.p):
            for j in range(i):
                if self.dim == self.p:
                    initq = self.R[i][j] # self.oldQ[i][j]
                else:
                    initq = self.R[i][j]
                localQ[i][j], iter = self.r2localq([self.mu[i], self.mu[j]], self.R[i][j], [self.mean[i], self.mean[j]], initq=initq)
                total_iter += iter
            localQ[0:i, i] = localQ[i, 0:i].T

        self.adjusted.append("%.3f average iter" % (total_iter / float(self.p * (self.p - 1) / 2.)))             
        self.adjusted.append("%.3f s" % (time.clock() - t))
        return localQ
    
    def r2localq(mu, rho, p=[inf, inf], initq=0, verbose=False):
        '''
        Computes the normal correlation q necessary to generate a bivariate bernoulli samples with correlation r.
        
            mu     mean of normal variates
            r      target correlation of bernoulli variates
            p      mean of bianry variates
            
        '''
        DECIMALS = 10. ** -5     # precision for breaking
        LOOPS = 50               # maximum number of loops
        
        for i in range(2):
            if abs(mu[i]) > 3.2: return 0., 0
            if p[i] == 'infty': p[i] = normal.cdf(mu[i])
          
        t = sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))

        # determine upper and lower bounds; make sure r is feasible
        bound = 0.99
        maxR = min(bound, (min(p[0], p[1]) - p[0] * p[1]) / t)
        minR = max(-bound, (max(p[0] + p[1] - 1, 0) - p[0] * p[1]) / t)
        rho = min(max(rho, minR), maxR)

        # set start value for Newton method
        q = initq

        # run Newton method
        lastq = inf; rerun = False
        s = p[0] * p[1] + rho * t
        for iter in range(LOOPS):
            if verbose == True: print i + 1, q               
            q = q - round((bvnormal.cdf(mu, q) - s), 8) / bvnormal.pdf(mu, q)
            # run binary search if Newton method fails for negative correlation
            if q < -1:
                if verbose == True: print "Newton method failed. Start binary search..."
                q = -0.5; u = 0; l = -1
                for iter in range(iter, LOOPS + iter):
                    v = (bvnormal.cdf(mu, q) - p[0] * p[1]) / t
                    if rho < v:
                        u = q; q = 0.5 * (q + l)
                    else:
                        l = q; q = 0.5 * (q + u)
                    # check precision bound
                    if abs(l - u) < DECIMALS: break
                return q, iter
            
            # set value to 0.999 if Newton step is greater 1
            if q > 1:
                if rerun == True: break # avoid endless loop
                rerun = True
                q = bound
            
            # check precision bound
            if abs(lastq - q) < DECIMALS: break
            lastq = q
        if q == inf or isnan(q): q = 0
        return max(min(q, 1.), -1.), iter
    r2localq = staticmethod(r2localq)

    def _pmf(self, gamma):
        raise ValueError("Evaluation of pmf is infeasible for binary_mn.")
       
    def _rvs(self):
        if self.p == 0: return
        
        # return sample from last step's multi normal
        if rand() < self.smooth_mean:
            v = normal_rand(size=len(self.oldmu))
            return self._expand_rv(rv=dot(self.oldC, v) < self.oldmu, previous=self.previous_strongly_random)
        else:
            v = normal_rand(size=self.p)
            return self._expand_rv(rv=dot(self.C, v) < self.mu)


class binary_log(binary):
    def __init__(self, data=None, fraction_mean=1, smooth_mean=0, fraction_corr=1, smooth_corr=0, threshold_randomness=0, \
                 min_p=0, verbose=False, weighted=False):
        '''
        Generates samples with independent components.
        
            data    a data object
            
            a       delay parameter
            b       fraction for estimation log regression
            c       mixing rate for independent samples
        '''
        
        binary.__init__(self, data=data, fraction_mean=fraction_mean, fraction_corr=fraction_corr, \
                        smooth_mean=smooth_mean, smooth_corr=smooth_corr, min_p=min_p, \
                        threshold_randomness=threshold_randomness, verbose=verbose)
        self.name = "binary_log"
                     
        # empty constructor
        if data == None or self.p < self.min_p: return

        # remember previous regression model
        if not hasattr(self, "previous_betas"):self.previous_betas = normal_rand(0, 0.01, (self.dim, self.dim))
        else: self.previous_betas = self.betas.copy()

        # gather data
        Xprime = self.data.data(fraction=fraction_corr, dtype=bool); n = len(Xprime)
        X = column_stack((ones(n, dtype=bool)[:, newaxis], Xprime))
        
        k = self.data.data(fraction=fraction_corr)

        self.betas = empty((self.p, self.p), dtype=float)
        self.betas[0][0] = sum(X[:, 1]) / float(n)

        if weighted: WX = self.data.weights(fraction=fraction_corr)[newaxis, :].T * X
        else: WX = X
        
        #
        #  loop over dimensions and compute logistic regressions
        #
        t = time.clock(); failures = 0; total_iter = 0
        index = [self.previous_strongly_random.index(j) for j in self.strongly_random if j in self.previous_strongly_random]
        
        # check for variables that were weakly random and became strongly random
        has_returnees = len(filter(lambda x:x not in self.previous_strongly_random, self.strongly_random)) > 0
        
        v = empty(n, dtype=float)
        
        for d in range(1, self.p):
            # use corresponding part of the sample
            y = X[:, d + 1]

            # init regression vector
            if has_returnees:
                self.betas[d][0:d + 1] = normal_rand(0, 0.01, d + 1)
            else:
                if self.dim == self.p:
                    self.betas[d][0:d + 1] = self.previous_betas[d][0:d + 1]
                else:
                    previous_d = self.previous_strongly_random.index(self.strongly_random[d])
                    self.betas[d][0:d + 1] = self.previous_betas[previous_d][index[:d + 1]]
            
            # Newton iterations
            for iter in range(1, LOOPS + 1):
                                    
                # compute probabilites 
                previous_beta = self.betas[d, :d + 1].copy()
                
                if WEAVE:
                    dprobs = empty(n);
                    beta = self.betas[d, :d + 1]
                    code = \
                    """
                    double p, Xbeta;
                    
                    for (int i = 0; i < n; i++)
                    {
                        Xbeta = 0;
                        for(int k = 0; k <= d; k++)
                        {
                            Xbeta += X(i,k) * beta(k);
                        }
                        p = 1 / (1 + exp(-Xbeta));
                        dprobs(i) = p * (1-p);
                        v(i) = dprobs(i) * Xbeta + y(i) - p;
                    }
                    """        
                    weave.inline(code, ['beta', 'X', 'y', 'dprobs', 'd', 'n', 'v'], \
                    type_converters=weave.converters.blitz, compiler='gcc')
                else:
                    Xbeta = dot(X[:, :d + 1], self.betas[d, :d + 1])
                    mean = pow(1 + exp(-Xbeta), -1)
                    dprobs = mean * (1 - mean)
                    v = dprobs * Xbeta + y - mean
                    
                WXDX = dot(WX[:, :d + 1].T, dprobs[newaxis, :].T * X[:, :d + 1]) + 10 ** -4 * eye(d + 1)
                # solve Newton equation
                self.betas[d, :d + 1] = solve(WXDX, dot(WX[:, :d + 1].T, v), sym_pos=True)
                # break iteration when precision is reached
                if (abs(previous_beta - self.betas[d, :d + 1]) < DECIMALS).all(): break
            
            # if Newton iteration did not converge, set conditional prob to marginal prob
            if iter == LOOPS:
                failures += 1
                self.betas[d, :d + 1] = zeros(d + 1)
                mprob = sum(X[:, d + 1]) / float(n)
                self.betas[d][0] = log(mprob / (1 - mprob))
            total_iter += iter

        # convergence statistics
        self.adjusted.append("%.3f average iter" % (total_iter / float(self.p - 1)))   
        self.adjusted.append("%i failures" % failures)          
        self.adjusted.append("%.3f s" % (time.clock() - t))
        if self.verbose: print self.adjusted
        
        
        
    def _pmf(self, gamma):
        return exp(lpmf(gamma))
  
    def lpmf(self, gamma):
        if WEAVE: partial_logprob = self._lpmf_weave(gamma[self.strongly_random])
        else: partial_logprob = self._lpmf_python(gamma[self.strongly_random])
        return self._expand_logprob(gamma, partial_logprob)

    def _lpmf_python(self, gamma):
        betas = self.betas; p = betas.shape[0]
        
        if gamma[0]: logvprob = log(betas[0][0])
        else: logvprob = log(1 - betas[0][0])
        
        # Compute log conditional probability that gamma(i) is one for i > 0
        sum = betas[1:, 0].copy()
        for i in range(1, p): sum[i - 1] += dot(betas[i, 1:i + 1], gamma[0:i])
        logcprob = -log(1 + exp(-sum))
        
        # Compute log conditional probability of whole gamma vector
        logvprob += logcprob.sum() - sum[-gamma[1:]].sum()
        return logvprob
    
    def _lpmf_weave(self, gamma):
        betas = self.betas; p = betas.shape[0]
        logvprob = empty(1, dtype=float)
        code = \
        """
        double sum, logcprob;
        int i,j;
               
        if (gamma(0)) logvprob = log(betas(0,0));
        else logvprob = log(1-betas(0,0));
        
        for(i=1; i<p; i++){
        
            /* Compute log conditional probability that gamma(i) is one */
            sum = betas(i,0);
            for(j=1; j<=i; j++){        
                sum += betas(i,j) * gamma(j-1);
            }
            logcprob = -log(1+exp(-sum));
            
            /* Compute log conditional probability of whole gamma vector */
            logvprob += logcprob;        
            if (!gamma(i)) logvprob -= sum;
            
        }
        """
        weave.inline(code, ['p', 'betas', 'gamma', 'logvprob'], \
                     type_converters=weave.converters.blitz, compiler='gcc')        
        return float(logvprob)
            
    def _rvs(self):
        if WEAVE: return self._expand_rv(self._rvs_weave()[0])
        else: return self._expand_rv(self._rvs_python()[0])
    
    def rvsplus(self):
        if WEAVE: rv, partial_logprob = self._rvs_weave()
        else: rv, partial_logprob = self._rvs_python()
        return self._expand_rv(rv, partial_logprob)
       
    def _rvs_python(self):
        if rand() < self.smooth_corr: return self.mean > rand(self.dim), 0
        if rand() < self.smooth_mean: betas = self.previous_betas
        else: betas = self.betas
        p = betas.shape[0]
        gamma = empty(p, dtype=bool)
        logu = log(random.rand(p))
        
        # Draw an independent gamma[0]
        gamma[0] = rand() < betas[0][0]
        if gamma[0]: logvprob = log(betas[0][0])
        else: logvprob = log(1 - betas[0][0])
        
        for i in range(1, p):            
            # Compute log conditional probability that gamma(i) is one
            sum = betas[i][0] + dot(betas[i, 1:i + 1], gamma[0:i])
            logcprob = -log(1 + exp(-sum))
            
            # Generate the ith entry
            gamma[i] = logu[i] < logcprob
            
            # Compute log conditional probability of whole gamma vector
            logvprob += logcprob
            if not gamma[i]: logvprob -= sum
        
        return gamma, logvprob

    def _rvs_weave(self):
        if rand() < self.smooth_corr: return self.mean > rand(self.dim), 0
        if rand() < self.smooth_mean: betas = self.previous_betas
        else: betas = self.betas
        
        p = betas.shape[0]
        u = random.rand(p)
        gamma = empty(p, dtype=bool)
        logvprob = empty(1, dtype=float)
        code = \
        """
        double sum, logcprob;
        int i,j;
        
        /* Draw an independent gamma(0) */
        gamma(0) = (u(0) < betas(0,0));
        
        if (gamma(0)) logvprob = log(betas(0,0));
        else logvprob = log(1-betas(0,0));
        
        for(i=1; i<p; i++){
        
            /* Compute log conditional probability that gamma(i) is one */
            sum = betas(i,0);
            for(j=1; j<=i; j++){        
                sum += betas(i,j) * gamma(j-1);
            }
            logcprob = -log(1+exp(-sum));
            
            /* Generate the ith entry */
            gamma(i) = (log(u(i)) < logcprob);
            
            /* Compute log conditional probability of whole gamma vector */
            logvprob += logcprob;        
            if (!gamma(i)) logvprob -= sum;
            
        }
        """
        weave.inline(code, ['p', 'u', 'betas', 'gamma', 'logvprob'], \
                     type_converters=weave.converters.blitz, compiler='gcc')
        return gamma, logvprob

class binary_post_old(binary):
    def __init__(self, target, variates, dataset='boston', scoretype='hb', intercept=True):
        '''
        Reads a dataset and construct the posterior probabilities of all linear models with variates regressed on target.
        
            target    the column to regress on
            data      vector of columns used as regressors
        '''
   
        self.dataset = dataset
        self.scoretype = scoretype
        # import dataset
        datreader = csv.reader(open('../../data/datasets/' + dataset + '/' + dataset + '.data'), delimiter=',', quotechar='|')
        # columns of explanatory variables
        for i, x in enumerate(variates):
            if type(x).__name__ == 'list':
                for j, y in enumerate(x):
                    variates[i][j] = y - 1
            else:
                variates[i] = x - 1
        self.variates = variates
        # column of explained variable
        self.target = target - 1
        # max number of observations
        self.n = inf
        # dimension
        self.p = len(self.variates)
        # add intercept
        if intercept: self.p += 1
        self.feasible = (self.p <= MAX_FEASIBLE)
        # init X and Y
        self.X = []; self.Y = []

        # store target and data names
        row = datreader.next()
        try:
            float(row[0])
            row = ['var_' + str(i) for i in range(1, len(row) + 1)]
            datreader = csv.reader(open('../../data/datasets/' + dataset + '/' + dataset + '.data'), delimiter=',', quotechar='|')
        except:
            pass
        self.names = []
        self.names.append(row[self.target])
        if intercept: self.names.append("1")
        for col in self.variates:
            if type(col) == types.ListType:
                colname = row[col[0]]
                for j in col[1:len(col)]:
                    colname = colname + "*" + row[j]
            else:
                colname = row[col]
            self.names.append(colname)
            
        index = 0

        if intercept: firstCol = 1
        else: firstCol = 0
        for row in datreader:
            if len(row) == 0: continue
            index += 1    
            # read explanatory variables
            set = ones(self.p)
            for i in range(firstCol, self.p):
                    col = self.variates[i - 1]
                    if type(col) == types.ListType:
                        try:
                            set[i] = float(row[col[0]])
                        except:
                            set[i] = 1.
                        for j in col[1:len(col)]:
                            try:
                                set[i] = set[i] * float(row[j])
                            except:
                                set[i] = set[i]
                    else:
                        try:
                            set[i] = float(row[col])
                        except:
                            set[i] = 1.                            
                        if i < self.p - 1:
                            if self.names[i + 1] == self.names[i + 2]:
                                set[i] += .01 * normal_rand()
                        if i > 0:
                            if self.names[i + 1] == self.names[i]:
                                set[i] += .01 * normal_rand()
            self.X.append(set)
            # read explained variables
            try:
                self.Y.append(float(row[self.target]))
            except:
                self.Y.append(1.)
            if index == self.n :break
        
        self.dim = self.p
        self.nozero = True
        self.name = "binary_post"
        self.title = "%r regressed on %r " % (self.names[0], self.names[1:len(self.names)])
        
        # compute XtY=beta and XtX
        self.n = index
        self.X = array(self.X)
        self.Y = array(self.Y)
        self.XtY = dot(self.X.T, self.Y)
        self.XtX = dot(self.X.T, self.X)
                
        '''
        parameterization
        '''
        if self.scoretype == 'bic':
            self.c1 = dot(self.Y.T, self.Y) / float(self.n)
            self.c2 = self.XtY / float(self.n)
            self.c3 = 0.5 * log(self.n)
        else:
            v1 = 100     # variance of beta
            lambda_ = 1  # inverse gamma prior of sigma^2
            nu_ = .1     # inverse gamma prior of sigma^2
            
            self.c1 = 0.5 * log(v1)
            self.c2 = 0.5 * (self.n + nu_)
            self.c3 = nu_ * lambda_ + dot(self.Y.T, self.Y)
            
        # explore the posterior if feasible
        if self.feasible: self.explore()
    
    def lpmf(self, gamma):
        '''
        Evaluates the unnormalized log pmf.
        
            gamma    a binary vector of length p
             
        '''
        if self.scoretype == "bic":
            return float(self.bic(gamma))
        else:
            return float(self.hb(gamma))
    
    def _pmf(self, gamma):
        '''
        Evaluates unnormalized pmf.
        
            gamma    a binary vector of length p
             
        '''
        if not self.feasible:
            raise ValueError('Cannot evaluate pmf for problem size %i.' % self.p)
        return exp(self.lpmf(gamma) - self.loglevel)
    
       
    def hb(self, gamma):
        '''
        Evaluates the log posterior density from a conjugate hierarchical setup (George, McCulloch, simplified).
        
            gamma    a binary vector of length p
             
        '''
        p = gamma.sum()
        if p == 0:
            return - inf
        else:
            K = cholesky(self.XtX[gamma, :][:, gamma] + 10 ** -8 * eye(p))
            if K.shape == (1, 1):
                w = self.XtY[gamma, :] / float(K)
            else:
                w = solve(K.T, self.XtY[gamma, :])
                '''
                Backward substitution - slower than scipy.linalg.solve - maybe inline ?
                
                b = self.XtY[gamma == 1, :].copy()
                w = zeros(p)
                for i in range(p):
                    w[i] = b[i] / K.T[i, i]
                    b[i + 1:] -= K.T[i + 1:, i] * w[i]
                '''
            k = log(K.diagonal()).sum()
        return - k - self.c1 * p - self.c2 * log(self.c3 - dot(w, w.T))
    
    def bic(self, gamma):
        '''
        Evaluates the Bayesian Information Criterion. 
        
            gamma    a binary vector of length p
             
        '''
        p = gamma.sum()
        if p == 0:
            return - inf
        else:
            try:
                beta = solve(self.XtX[gamma, :][:, gamma], self.XtY[gamma, :], sym_pos=True)
            except:
                beta = solve(self.XtX[gamma, :][:, gamma] + 10 ** -8 * eye(p), self.XtY[gamma, :], sym_pos=True)
        return - 0.5 * self.n * log(self.c1 - dot(self.c2[gamma], beta)) - self.c3 * p
    
    def explore(self):
        '''
        Explores the posterior for feasible dimensions.
        '''
        self.loglevel = -inf
        # find the maximum of the pmf needed for acception/rejection sampling
        for d in range(2 ** self.p):
            b = dec2bin(d, self.p)
            eval = self.lpmf(b)
            if eval > self.loglevel: self.loglevel = eval
   
    def _rvs(self):
        '''
        Generates a sample from the posterior density
        rejecting from a proposals of p independent 0.5-binary variables.
        
        If p > MAX_FEASIBLE the method returns a zero vector.    
        '''
        if not self.feasible:
            raise ValueError('Cannot evaluate pmf for problem size %i.' % self.p)
        while True:
            proposal = 0.5 * ones(self.p) > rand(self.p)
            if rand() < self.pmf(proposal): return proposal













class binary_post(binary):
    def __init__(self, dataset, scoretype='hb'):
        '''
        Reads a dataset and construct the posterior probabilities of all linear models
        with variates regressed on the first column.
        '''
   
        # import dataset
        X = []; Y = []
        datreader = csv.reader(open(dataset), delimiter=';')
        for row in datreader:
            X.append(array([float(entry) for entry in row[1:]]))
            Y.append(float(row[0]))
        
        self.name = "binary_post"
        self.scoretype=scoretype
        self.dim = len(X[0])
        self.feasible = (self.dim <= MAX_FEASIBLE)
        self.n = len(X)
        
        # compute beta=XtY and XtX
        X = array(X); Y = array(Y)
        self.XtY = dot(X.T, Y)
        self.XtX = dot(X.T, X)
                
        # parameterization
        if self.scoretype == 'hb':
            v1 = 100     # variance of beta
            lambda_ = 1  # inverse gamma prior of sigma^2
            nu_ = .1     # inverse gamma prior of sigma^2
            self.c1 = 0.5 * log(v1)
            self.c2 = 0.5 * (self.n + nu_)
            self.c3 = nu_ * lambda_ + dot(Y.T, Y)
            
        if self.scoretype == 'bic':
            self.c1 = dot(Y.T, Y) / float(self.n)
            self.c2 = self.XtY / float(self.n)
            self.c3 = 0.5 * log(self.n)
            
        # explore the posterior if feasible
        if self.feasible: self._explore()
    
    def lpmf(self, gamma):
        '''
        Evaluates the unnormalized log pmf.
        
            gamma    a binary vector
        '''
        if self.scoretype == "bic":
            return float(self.bic(gamma))
        else:
            return float(self.hb(gamma))
    
    def _pmf(self, gamma):
        '''
        Evaluates unnormalized pmf.
        
            gamma    a binary vector of length p
             
        '''
        if not self.feasible:
            raise ValueError('Cannot evaluate pmf for problem size %i.' % self.p)
        return exp(self.lpmf(gamma) - self.loglevel)

    def hb(self, gamma):
        '''
        Evaluates the log posterior density from a conjugate hierarchical setup ([George, McCulloch 1997], simplified).
        
            gamma    a binary vector
        '''
        p = gamma.sum()
        if p == 0:
            return - inf
        else:
            K = cholesky(self.XtX[gamma, :][:, gamma] + 10 ** -8 * eye(p))
            if K.shape == (1, 1):
                w = self.XtY[gamma, :] / float(K)
            else:
                w = solve(K.T, self.XtY[gamma, :])
            k = log(K.diagonal()).sum()
        return - k - self.c1 * p - self.c2 * log(self.c3 - dot(w, w.T))
    
    def bic(self, gamma):
        '''
        Evaluates the Bayesian Information Criterion. 
        
            gamma    a binary vector
        '''
        p = gamma.sum()
        if p == 0:
            return - inf
        else:
            try:
                beta = solve(self.XtX[gamma, :][:, gamma], self.XtY[gamma, :], sym_pos=True)
            except:
                beta = solve(self.XtX[gamma, :][:, gamma] + 10 ** -8 * eye(p), self.XtY[gamma, :], sym_pos=True)
        return - 0.5 * self.n * log(self.c1 - dot(self.c2[gamma], beta)) - self.c3 * p
    
    def _explore(self):
        '''
        Find the maximmum of the log posterior for feasible dimensions.
        '''
        self.loglevel = -inf
        for d in range(2 ** self.p):
            b = dec2bin(d, self.p)
            eval = self.lpmf(b)
            if eval > self.loglevel: self.loglevel = eval
   
    def _rvs(self):
        '''
        Generates a sample from the posterior density rejecting from a proposals
        of independent 0.5-binary variables.   
        '''
        if not self.feasible:
            raise ValueError('Cannot evaluate pmf for problem size %i.' % self.p)
        while True:
            proposal = 0.5 * ones(self.dim) > rand(self.dim)
            if rand() < self.pmf(proposal): return proposal

