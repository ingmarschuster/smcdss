'''
@author cschafer
'''

from auxpy.data import *
from numpy import array, ones, zeros, log
from numpy.random import rand
from scipy.stats import rv_discrete

class product_binary(rv_discrete):
    '''
        A multivariate Bernoulli with independent components.
    '''
    def __init__(self, p=None, name='product-binary', longname='A multivariate Bernoulli with independent components.'):
        '''
            Constructor.
            @param p mean vector
            @param name name
            @param longname longname
        '''
        rv_discrete.__init__(self, name=name, longname=longname)
        if not p==None:
            try:
                ## mean vector
                self.p = array(p, dtype=float)
                ## dimension
                self.d = p.shape[0]
            except:
                self.p = array([p])
                self.d = 1

    def pmf(self, gamma):
        '''
            Probability mass function.
            @param gamma: binary vector
        '''
        prob = 1.
        for i, mprob in enumerate(self.p):
            if gamma[i]: prob *= mprob
            else: prob *= (1 - mprob)
        return prob

    def lpmf(self, gamma):
        '''
            Log-probability mass function.
            @param gamma binary vector
        '''
        return log(self.pmf(gamma))

    def rvs(self):
        '''
            Generates a random variate.
        '''
        if self.d == 0: return []
        return self.p > rand(self.d)

    def rvslpmf(self):
        '''
            Generates a random variate and computes its likelihood.
        '''
        rv = self.rvs()
        return rv, self.lpmf(rv)

    def rvstest(self, n):
        '''
            Compares the empirical and the true mean.
            @param n sample size
        '''
        sample = data()
        sample.sample(self, n)
        print str(self)
        print format(sample.mean,'sample mean')

    def __str__(self):
        return format_vector(self.p, 'p')


    @classmethod
    def random(cls, d):
        '''
            Construct a random product-binary model for testing.
            @param cls class
            @param d dimension
        '''
        return cls(rand(d))

    @classmethod
    def uniform(cls, d):
        '''
            Construct a random product-binary model for testing.
            @param cls class
            @param d dimension
        '''
        return cls(0.5 * ones(d))
    
    @classmethod
    def fromData(cls, sample):
        '''
            Construct a product-binary model from data.
            @param cls class
            @param sample a sample of binary data
        '''
        return cls(sample.mean)
