#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from copy import deepcopy
from time import clock
from pickle import dump, load

from sys import stdout
from numpy import *

class data(object):
    def __init__(self, X=[], w=[]):
        '''
            Data class.
            @param X data
            @param w weights  
        '''

        ## data
        self._X = list(X)
        ## weights
        self._W = list(w)
        ## index set
        self.__order = None

    def __str__(self):
        return format(self.getMean(weight=True), 'mean') + '\n' + \
               format(self.getCor(weight=True), 'correlation')

    def fraction(self, fraction=1.0):
        return data(X=self.X[self.order[0:int(self.size * fraction)]], \
                    w=array(self._W)[self.order[0:int(self.size * fraction)]])

    def getData(self):
        '''
            Get data.
            @return data array.
        '''
        if isinstance(self._X, list):
            return array(self._X)
        else:
            return self._X

    def proc_data(self, order=False, fraction=1.0, dtype=int):
        '''
            Get processed data.
            @param order weights in ascending order
            @param fraction upper fraction of the ordered data
            @param dtype data type
            @return data array
        '''
        if fraction == 1.0:
            if order:
                return array(self.X[self.order], dtype=dtype)
            else:
                return array(self.X, dtype=dtype)
        else:
            return array(self.X[self.order[0:int(self.size * fraction)]], dtype=dtype)

    def getNWeights(self):
        '''
            Get weights.
            @remark If weights are negative, the function returns the normalized exponential weights.
            @return normalized weights
        '''
        if not self.isWeighted(): return ones(self.size) / float(self.size)
        w = array(self._W)
        max = w.max()
        if max < 0: w = exp(self._W - max)
        return w / w.sum()

    def getWeights(self, normalized=False):
        return self._W

    def proc_weights(self, order=False, fraction=1.0):
        '''
            Get processed weights.
            @param order weights in ascending order
            @param fraction upper fraction of the ordered weights
            @return weights
        '''
        if fraction == 1.0:
            if order:
                return self.nW[self.order]
            else:
                return self.nW
        else:
            w = self.nW[self.order[0:int(self.size * fraction)]]
            return w / w.sum()

    def clear(self, fraction=1.0):
        '''
            Deletes the data.
            @param fraction keep upper fraction of the ordered data
        '''
        if fraction == 1.0:
            self.__init__()
        else:
            self.__init__(X=self.X[self.order[0:int(self.size * fraction)]], \
                          w=array(self._W)[self.order[0:int(self.size * fraction)]])

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        if size == 0: return 0
        return len(self._X[0])

    def getSize(self):
        '''
            Get sample size.
            @return sample size 
        '''
        return len(self._X)

    def setData(self, X):
        '''
            Set data.
            @param X data
        '''
        self._X = list(X)

    def setWeights(self, w):
        '''
            Set weights.
            @param w weights
        '''
        self._W = list(w)

    def isWeighted(self):
        '''
            Test if weighted.
            @return True, if the sample has weights.
        '''
        return len(self._W) > 0

    def getMean(self, weight=False, fraction=1.0):
        '''
            Computes the mean.
            @param weight compute weighted mean
            @param fraction use only upper fraction of the ordered data
        '''
        if weight:
            return calc_mean(self.proc_data(fraction=fraction, dtype=float), w=self.proc_weights(fraction=fraction))
        else:
            return calc_mean(self.proc_data(fraction=fraction, dtype=float))

    def getCov(self, weight=False, fraction=1.0):
        '''
            Computes the covariance matrix.
            @param weight compute weighted covariance
            @param fraction use only upper fraction of the ordered data
        '''
        if weight:
            return calc_cov(X=self.proc_data(fraction=fraction, dtype=float), w=self.proc_weights(fraction=fraction))
        else:
            return calc_cov(X=self.proc_data(fraction=fraction, dtype=float))

    def getCor(self, weight=False, fraction=1.0):
        '''
            Computes the correlation matrix.
            @param weight compute weighted correlation
            @param fraction use only upper fraction of the ordered data
        '''
        if weight:
            return calc_cor(X=self.proc_data(fraction=fraction, dtype=float), w=self.proc_weights(fraction=fraction))
        else:
            return calc_cor(X=self.proc_data(fraction=fraction, dtype=float))

    def getVar(self, weight=False, fraction=1.0):
        '''
            Computes the variance vector.
            @param weight compute weighted variance
            @param fraction use only upper fraction of the ordered data
        '''
        return diag(self.getCov(weight=weight, fraction=fraction))

    def getEss(self):
        '''
        Return effective sample size 1/(sum_{w \in weights} w^2) .
        '''
        if not self.isWeighted(): return 0
        return 1 / (self.size * pow(self.nW, 2).sum())

    def getOrder(self):
        '''
            Get order.
            @return index set for the data in ascending order according to the weights.
        '''
        if self.__order is None: self.__sort()
        if self.__order is None: self.__order = range(size)
        return self.__order

    def __sort(self):
        '''
            Sets the index for the data in ascending order according to the weights.
        '''
        self.__order = array(self._W).argsort(axis=0).tolist()
        self.__order.reverse()


    def lexorder(self, X):
        '''
            Gives the index for the data in lexicographical order.
            @return index set 
        '''
        #return argsort(array([str(array(x, int)) for x in X]))
        k = array([2 ** i for i in range(len(X[0]))])
        return argsort(array([dot(k, x) for x in X]))

    def assign_weights(self, f, verbose=False):
        '''
            Evaluates the value of c*exp(f(x)) for each sample x.
            @param f a real-valued function on the sampling space 
        '''

        if verbose:
            t = clock()
            bars = 20
            drawn = 0
            print 'Evaluating ' + f.name + '...'
            stdout.write('[' + bars * ' ' + "]" + "\r" + "[")

        v = self._W
        k = len(v)
        X = array(self._X)[k:]
        self.nW = []

        # Apply in lexicographical order to avoid extra evaluation of f.
        lexorder = self.lexorder(X)
        weight = f.lpmf(X[lexorder[0]])
        for index in range(self.size - k):
            if not (X[lexorder[index]] == X[lexorder[index - 1]]).all():
                weight = f.lpmf(X[lexorder[index]])
            self._W.append(weight)
            if verbose:
                n = bars * (index + 1) / (self.size - k) - drawn
                if n > 0:
                    stdout.write(n * "-")
                    stdout.flush()
                    drawn += n
        self._W = array(self._W)
        self._W = v + list(self._W[argsort(lexorder)])

        if verbose: print ']\nDone. %i evaluation in %.2f seconds.\n' % (self.size - k, clock() - t)

    def distinct(self):
        X = self._X
        W = self.nW
        lexorder = self.lexorder(X)
        self._X = []; self._W = []
        x = X[lexorder[0]]
        count = 1
        for index in lexorder:
            if (x == X[index]).all():
                count += 1
            else:
                self._X += [x]
                self._W += [W[index] * count]
                x = X[index]
                w = W[index]
                count = 1

    def append(self, x, w=None):
        '''
            Appends the value x with weigth w to the dataset.
            @param x value
            @param w weight
        '''
        if not isinstance(self._X, list): self._X.tolist()
        if not isinstance(self._W, list): self._W.tolist()
        self._X.append(x)
        if not w is None: self._W.append(float(w))

    def shrink(self, index):
        '''
            Removes the indicated columns from the data.
            @param index column index set 
        '''
        self._X = self.X[:, index]

    def get_sub_data(self, index):
        '''
            Returns a shrinked version of the data class.
            @param index column index set 
        '''
        data = self.copy()
        data.shrink(index)
        return data

    def copy(self):
        '''
            Creates a deep copy.
            @return deep copy
        '''
        return deepcopy(self)

    def sample(self, q, size, verbose=False):
        '''
            Samples from a random generator.
            @param q random generator
            @param size sample size
            @param verbose print status line
        '''
        if verbose:
            t = clock()
            bars = 20
            drawn = 0
            print 'Sampling from ' + q.name + '...'
            stdout.write('[' + bars * ' ' + "]" + "\r" + "[")
        for i in range(1, size + 1):
            self.append(q.rvs())
            if verbose:
                n = bars * i / size - drawn
                if n > 0:
                    stdout.write(n * "-")
                    stdout.flush()
                    drawn += n
        if verbose: print ']\nDone. %i variables sampled in %.2f seconds.\n' % (size, clock() - t)

    def save(self, filename):
        '''
            Saves the sample to a file using pickle.
            @param filename filename
        '''
        dump(self, open(filename, 'wb'))

    def load(self, filename):
        '''
            Loads a sample from a file using pickle.
            @param filename filename
        '''
        data = load(open(filename))
        self.__init__(data.X, data.w)

    X = property(fget=getData, fset=setData, doc="data")
    nW = property(fget=getNWeights, fset=setWeights, doc="normalized weights")
    W = property(fget=getWeights, fset=setWeights, doc="weights")
    ess = property(fget=getEss, doc="effective sample size")
    mean = property(fget=getMean, doc="mean")
    var = property(fget=getVar, doc="variance")
    cov = property(fget=getCov, doc="covariance matrix")
    cor = property(fget=getCor, doc="correlation matrix")
    d = property(fget=getD, doc="dimension")
    size = property(fget=getSize, doc="sample size")
    order = property(fget=getOrder, doc="index of data in ascending order according to the weights")


def calc_mean(X, w=None):
    '''
        Mean.
        @param X array
        @param w positive weights
        @return mean
    '''
    if w is None:
        return X.sum(axis=0) / float(X.shape[0])
    else:
        return (w[:, newaxis] * X).sum(axis=0)

def calc_cov(X, w=None):
    '''
        Covariance.
        @param X array
        @param w positive weights
        @return covariance matrix
    '''
    if w is None:
        n = float(X.shape[0])
        mean = calc_mean(X)[newaxis, :]
        return (dot(X.T, X) - n * dot(mean.T, mean)) / float(n - 1)
    else:
        mean = calc_mean(X, w)[newaxis, :]
        return (dot(X.T, w[:, newaxis] * X) - dot(mean.T, mean)) / (1 - pow(w, 2).sum())

def calc_cor(X, w=None):
    '''
        Correlation.
        @param X array
        @param w positive weights
        @return correlation matrix
    '''
    d = X.shape[1]
    cov = calc_cov(X, w) + exp(-10) * eye(d)
    var = cov.diagonal()[newaxis, :]
    return cov / sqrt(dot(var.T, var))

def calc_norm(v, p=2.0):
    return pow(pow(abs(v), p).sum(axis=0), 1.0 / p)

def format(X, name=''):
    '''
        Formats a vector or matrix for output on stdout
        @param X vector or matrix
        @param name name 
    '''
    if len(X.shape) == 1: return format_vector(X, name)
    if len(X.shape) == 2: return format_matrix(X, name)

def format_vector(v, name=''):
    '''
        Formats a vector for output on stdout
        @param v vector 
        @param name name 
    '''
    if not name == '': name = name + ' =\n'
    return name + '[' + ' '.join([('%.4f' % x).rjust(7) for x in v]) + ' ]\n'

def format_matrix(M, name=''):
    '''
        Formats a matrix for output on stdout
        @param M matrix
        @param name name 
    '''
    if not name == '': name = name + ' =\n'
    return name + ''.join([format_vector(x) for x in M])

def bin2str(bin):
    '''
        Converts a boolean array to a string representation.
        @param bin boolean array 
    '''
    return ''.join([str(i) for i in array(bin, dtype=int)])

def bin2dec(bin):
    '''
        Converts a boolean array into an integer.
        @param bin boolean array 
    '''
    return long(bin2str(bin), 2)

def dec2bin(n, d=0):
    '''
        Converts an integer into a boolean array containing its binary representation.
        @param n integer
        @param d dimension of boolean vector
    '''
    bin = []
    while n > 0:
        if n % 2: bin.append(True)
        else: bin.append(False)
        n = n >> 1
    while len(bin) < d: bin.append(False)
    bin.reverse()
    return array(bin)
