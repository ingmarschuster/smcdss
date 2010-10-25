'''
@author: cschafer
'''

from numpy import array, ones, zeros
from numpy.random import rand
from scipy.stats import rv_discrete

class product(rv_discrete):
    '''
    Product-binary model.
    '''
    def __init__(self, p):
        rv_discrete.__init__(self, name='product-bernoulli')
        try:
            self.p = array(p, dtype=float)
        except:
            raise ValueError('The argument is neither list nor array.')
        self.d = len(p)

    def pmf(self, gamma):
        prob = 1.
        for i, mprob in enumerate(self.p):
            if gamma[i]: prob *= mprob
            else: prob *= (1 - mprob)
        return prob
    
    def lpmf(self, gamma):
        return log(pmf(gamma))

    def rvs(self):
        if self.d == 0: return []
        return self.p > rand(self.d)

    def rvslpmf(self):
        rv = self._rvs()
        return rv, self.lpmf(rv)

    def autotest(self, n):
        print 'true p '.ljust(10)+ ': ' + ' '.join(['%.3f' % x for x in self.p])
        sum = zeros(self.d)
        for i in range(n):
            sum += array(self.rvs(), dtype=float)
        print 'average'.ljust(10) + ': ' + ' '.join(['%.3f' % x for x in sum / n])


class rproduct(product):
    '''
    Initializes a random product-binary model.
    '''
    def __init__(self, d):
        product.__init__(self, p=rand(d))
    
class uproduct(product):
    '''
    Initializes a uniform binary model.
    '''
    def __init__(self, d):
        product.__init__(self, p=.5 * ones(d))

    def pmf(self, gamma):
        return 2 ** -self.d
    
    def lpmf(self, gamma):
        return - self.d * log(2)
