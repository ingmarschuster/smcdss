'''
@author: cschafer
'''

from binary.product import *
from numpy import *
from scipy.linalg import cholesky, eigvalsh, solve
from scipy.stats import norm 

class normal():
    def __init__(self, mu, Q):
        self.mu = mu
        self.Q = Q
        try:
            self.C = cholesky(Q, True)
        except:
            print "Q is not positive definite. Set Q to identity."
            self.C = eye(self.d)
        
    def pmf(self, gamma):
        raise ValueError("No evaluation of the pmf for the normal-binary model.")

    def lpmf(self, gamma):
        raise ValueError("No evaluation of the pmf for the normal-binary model.")
       
    def rvs(self):
        if self.d == 0: return
        v = random.normal(size=self.d)
        return dot(self.C, v) < self.mu
    
    def autotest(self, n):
        print 'true p '.ljust(10) + ': ' + ' '.join(['%.3f' % x for x in self.p])
        sum = zeros(self.d)
        for i in range(n):
            sum += array(self.rvs(), dtype=float)
        print 'average'.ljust(10) + ': ' + ' '.join(['%.3f' % x for x in sum / n])

class rnormal(rproduct, normal):
    def __init__(self, d):
        
        ''' Init random product-binary.'''
        rproduct.__init__(self, d)
        mu = norm.ppf(self.p)
        
        ''' For a random matrix X with entries U[-1,1], set Q = X*X^t and normalize.'''  
        X = ones((d, d)) - 2 * random.random((d, d))
        Q = dot(X, X.T) + exp(-10) * eye(d)
        q = Q.diagonal()[newaxis, :]
        Q = Q / sqrt(dot(q.T, q))
        
        normal.__init__(self, mu, Q)
