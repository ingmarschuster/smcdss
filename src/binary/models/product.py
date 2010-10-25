'''
Created on 19 oct. 2010

@author: cschafer
'''

import binary.models.binary

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