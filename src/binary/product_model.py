'''
@author: cschafer
'''

from auxpy.format import *
from numpy import array, ones, zeros, log
from numpy.random import rand
from scipy.stats import rv_discrete

class product_binary(rv_discrete):
    '''
    Product-binary model.
    '''
    def __init__(self, p):
        rv_discrete.__init__(self, name='product-binary')
        try:
            self.d = len(p)
            self.p = array(p, dtype=float)
        except:
            self.d = 1
            self.p = array([p])

    def pmf(self, gamma):
        ''' Probability mass function. '''
        prob = 1.
        for i, mprob in enumerate(self.p):
            if gamma[i]: prob *= mprob
            else: prob *= (1 - mprob)
        return prob

    def lpmf(self, gamma):
        ''' Log-probability mass function. '''
        return log(self.pmf(gamma))

    def rvs(self):
        ''' Generates a random variate. '''
        if self.d == 0: return []
        return self.p > rand(self.d)

    def rvslpmf(self):
        ''' Generates a random variate and computes its likelihood. '''
        rv = self.rvs()
        return rv, self.lpmf(rv)

    def autotest(self, n):
        ''' Generates a sample of size n. Compares the empirical and the true mean. '''
        mean = zeros(self.d)
        for i in range(n):
            mean += array(self.rvs(), dtype=float)
        print 'true p\n' + format_vector(self.p)
        print 'average\n' + format_vector(mean / n)


    @classmethod
    def random(cls, d):
        ''' Construct a random product-binary model for testing. '''
        return cls(rand(d))

    @classmethod
    def uniform(cls, d):
        ''' Construct a random product-binary model for testing.'''
        return cls(0.5 * ones(d))
