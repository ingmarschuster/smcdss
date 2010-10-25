'''
Created on 19 oct. 2010

@author: cschafer
'''



class genericBinary(rv_discrete):
    '''
    The generator interface to be implemented by generators.
    '''
    def __init__(self, name):
        rv_discrete.__init__(self, name=name)
        
    def pmf(self, gamma):
        '''
        Evaluate probability mass function.
        '''  
        return self._pmf(gamma)
    
    def lpmf(self, gamma):
        '''
        Evaluate log probability mass function.
        '''  
        return log(self._pmf(gamma))

    def rvs(self):
        '''
        Generate a random variable.
        '''       
        return self._rvs()
    
    def rvslpmf(self):
        '''
        Generates a random variable and evaluates the unnormalized log-likelihood.
        '''       
        return self._rvslpmf()
    
        