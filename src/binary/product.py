'''
Created on 19 oct. 2010

@author: cschafer
'''

class productBinary(rv_discrete):
    '''
    Product-binary model.
    '''
    def __init__(p):
        rv_discrete.__init__(self, name='product-binary')

    def pmf(self, gamma):
        if self.uniform > 0: return self.uniform
        prob = 1.
        for i, mprob in enumerate(self.mean):
            if gamma[i]: prob *= mprob
            else: prob *= (1 - mprob)
        return prob
    
    def lpmf(self, gamma):
        return log(pmf(gamma))

    def rvs(self):
        if self.p == 0: return []
        rv = self.mean[self.strongly_random] > rand(self.p)
        return self._expand_rv(rv)

    def rvslpmf(self):
        rv = self._rvs()
        return self._expand_rv(rv, self.lpmf(rv))


class rproductBinary(productBinary):
    '''
    Initializes a random product-binary model.
    '''
    def __init__(d):
        binary.__init__(self, p=random.rand(d))
    
class uproductBinary(productBinary):
    '''
    Initializes a uniform binary model.
    '''
    def __init__(d):
        binary.__init__(self, p=.5 * ones(d))
        

