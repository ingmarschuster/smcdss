'''
Created on 29 oct. 2009

@author: cschafer
'''

from sys import stdout
from numpy import *
import time, copy

class data(object):
    def __init__(self, dataset=None, index=None, weightset=None):
        '''
        The data class holds a list of random samples. 
        '''
        if dataset == None:
            self.dataset = []
            self.weightset = []
        else:
            if index == None:
                self.dataset = dataset
                self.weightset = weightset
            else:
                self.dataset = dataset[:, index]
                self.weightset = weightset[:, index]
        self.isArray = False
        self.nweightset = None
        self.order = None
        self.index = None

    def append(self, sample, weight=None):
        self.index = None
        if self.isArray: self.tolist()
        self.dataset.append(sample)
        if not weight == None: self.weightset.append(float(weight))
        
    def toarray(self):
        if self.isArray: return
        self.dataset = array(self.dataset)
        if not len(self.weightset) == 0:
            self.weightset = array(self.weightset)
        self.isArray = True
    def tolist(self):
        if not self.isArray: return
        self.dataset = self.dataset.tolist()
        if not len(self.weightset) == 0:
            self.weightset = self.weightset.tolist()
        self.isArray = False
        self.order = None
    def data(self, ordered=False, fraction=1, dtype=int):
        self.toarray()
        if self.index == None:
            if fraction == 1:
                if ordered:
                    if self.order == None: self.sort()
                    return array(self.dataset[self.order], dtype=dtype)
                else:
                    return array(self.dataset, dtype=dtype)
            else:
                if self.order == None: self.sort()
                return array(self.dataset[self.order[0:int(self.size() * fraction)]], dtype=dtype)
        else:
            if fraction == 1:
                if ordered:
                    if self.order == None: self.sort()
                    return array(self.dataset[self.order, :][:, self.index], dtype=dtype)
                else:
                    return array(self.dataset[:, self.index], dtype=dtype)
            else:
                if self.order == None: self.sort()
                return array(self.dataset[self.order[0:int(self.size() * fraction)], :][:, self.index], dtype=dtype)
    def weights(self, ordered=False, fraction=1):
        if len(self.weightset) == 0:return []
        self.toarray()
        if fraction == 1:
            if ordered:
                if self.order == None: self.sort()
                w = self.weightset[self.order]
            else:
                w = self.weightset
        else:
            if self.order == None: self.sort()
            w = self.weightset[self.order[0:int(self.size() * fraction)]]
        return w
    def clear(self, fraction=1):
        if fraction == 1:
            self.__init__()
        else:
            self.sort()
            self.__init__(dataset=self.dataset[self.order[0:int(self.size() * fraction)]], \
                          weightset=self.weightset[self.order[0:int(self.size() * fraction)]])
            self.isArray = True
            self.tolist()
    def size(self):
        return len(self.dataset)
    def dim(self):
        if self.index == None:
            return len(self.dataset[0])
        else:
            return len(self.index)         
    def cov(self, weighted=False, indexed=False, fraction=1):
        self.toarray()
        if weighted:
            if len(self.weightset) == 0:
                print "WARNING: No weights assigned. Return unweighted cov."
                return self.cov()
            w = self.weights(fraction=fraction)[newaxis, :]
            m = self.mean(weighted=weighted, fraction=fraction)[newaxis, :]
            Q = dot(self.data(fraction=fraction).T, w.T * self.data(fraction=fraction)) - dot(m.T, m)
            Q /= 1 - pow(w, 2).sum()
        else:
            n = int(fraction * self.size())
            m = self.mean(fraction=fraction)[newaxis, :]
            Q = (dot(self.data(fraction=fraction).T, self.data(fraction=fraction)) - n * dot(m.T, m)) / (n - 1)
        return Q
    def cor(self, weighted=False, fraction=1):
        # ensure the matrix has full rank
        Q = self.cov(weighted=weighted, fraction=fraction) + 10 ** -5 * eye(self.dim())
        q = Q.diagonal()[newaxis, :]
        Q /= sqrt(dot(q.T, q))
        return Q
    def mean(self, weighted=False, indexed=False, fraction=1, dtype=int):
        self.toarray()
        if weighted:
            if len(self.weightset) == 0:
                print "WARNING: No weights assigned. Return unweighted mean."
                return self.mean()
            return (self.weights(fraction=fraction)[:, newaxis] * self.data(fraction=fraction, dtype=dtype)).sum(axis=0)
        else:
            return self.data(fraction=fraction, dtype=dtype).sum(axis=0) / float(int(fraction * self.size()))
    def setweights(self, post):
        evals = 1
        models = []
        for i in range(self.size()):
            models.append(''.join(map(lambda x: str(int(x)), self.dataset[i])))
        index = argsort(models)
        self.weightset = []
        weight = post.lpmf(array(self.dataset[0]))
        for i in index:
            if not models[i] == models[i - 1]:
                weight = post.lpmf(array(self.dataset[i]))
                evals += 1
            self.weightset.append(weight)
        self.toarray()
        self.weightset = array(self.weightset)[argsort(index)]
        return evals
        
    def sort(self):
        self.toarray()
        if len(self.weights()) == 0:
            print "WARNING: No weights assigned."
            return
        self.order = self.weightset.argsort(axis=0).tolist()
        self.order.reverse()
    def shrink(self, index):
        if not self.isArray: self.toarray()
        self.dataset = self.dataset[:, index]
    def copy(self):
        return copy.deepcopy(self)

class sampler(object):
    def __init__(self, gen, targetDistr=None):
        '''
        The sampler class holds a data object and fills it with random samples from a given generator.
        
        generator    a random number generator
        '''
        self.gen = gen
        self.data = data()
        self.targetDistr = targetDistr

    def sample(self, size, verbose=False, online=False):
        if verbose:
            print "sampling from " + self.gen.name
            step = size / 40.
            start = time.clock()
            print "[",
        evals = 0
        for i in range(0, size):
            if verbose:
                if i % step == 0:
                    stdout.write("-")
                    stdout.flush()
            b = self.gen.rvs()
            self.data.append(b)
            if not self.targetDistr == None and online:
                self.data.weightset.append(self.targetDistr.lpmf(b))
        self.data.toarray()
        if not self.targetDistr == None and not online:
            evals = self.data.setweights(self.targetDistr)
        else:
            evals = size
        if verbose:
            print "]"
            print "sampling finished in %.2f seconds." % (time.clock() - start)
        return evals

