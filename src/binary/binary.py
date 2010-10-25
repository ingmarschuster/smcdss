'''
Created on 19 oct. 2010

@author: cschafer
'''

from ..auxiliary import decbin

class binary(rv_discrete):
    '''
    The generator interface to be implemented by generators.
    '''
    def __init__(self, verbose=False):
        self.verbose=False
        rv_discrete.__init__(self, name='binary')
            
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
        