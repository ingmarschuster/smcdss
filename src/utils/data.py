#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Data processing and sampling. \utils.data """

import copy
import numpy
import pickle

class data(object):
    def __init__(self, X=[], w=[]):
        """ Data class.
            \param X data
            \param w weights  
        """

        ## data
        self._X = list(X)
        ## weights
        self._W = list(w)
        ## index set
        self.__order = None

    def __str__(self):
        return ('sum =\n%.3f \n' % numpy.exp(self._W).sum()) + \
               format(self.getMean(weight=True), 'mean') + \
               format(self.getCor(weight=True), 'correlation')

    def fraction(self, fraction=1.0):
        return data(X=self.X[self.order[0:int(self.size * fraction)]], \
                    w=numpy.array(self._W)[self.order[0:int(self.size * fraction)]])

    def getData(self):
        """
            Get data.
            \return data array.
        """
        if isinstance(self._X, list):
            return numpy.array(self._X)
        else:
            return self._X

    def proc_data(self, order=False, fraction=1.0, dtype=int):
        """
            Get processed data.
            \param order weights in ascending order
            \param fraction upper fraction of the ordered data
            \param dtype data type
            \return data array
        """
        if fraction == 1.0:
            if order:
                return numpy.array(self.X[self.order], dtype=dtype)
            else:
                return numpy.array(self.X, dtype=dtype)
        else:
            return numpy.array(self.X[self.order[0:int(self.size * fraction)]], dtype=dtype)

    def getNWeights(self):
        """
            Get weights.
            @remark If weights are negative, the function returns the normalized exponential weights.
            \return normalized weights
        """
        if not self.isWeighted(): return numpy.ones(self.size) / float(self.size)
        w = numpy.array(self._W)
        if w.min() < 0.0: w = numpy.exp(self._W - w.max())
        w = w / w.sum()
        return w / w.sum()

    def getWeights(self, normalized=False):
        return self._W

    def proc_weights(self, order=False, fraction=1.0):
        """
            Get processed weights.
            \param order weights in ascending order
            \param fraction upper fraction of the ordered weights
            \return weights
        """
        if fraction == 1.0:
            if order:
                return self.nW[self.order]
            else:
                return self.nW
        else:
            w = self.nW[self.order[0:int(self.size * fraction)]]
            return w / w.sum()

    def clear(self, fraction=1.0):
        """
            Deletes the data.
            \param fraction keep upper fraction of the ordered data
        """
        if fraction == 1.0:
            self.__init__()
        else:
            self.__init__(X=self.X[self.order[0:int(self.size * fraction)]], \
                          w=numpy.array(self._W)[self.order[0:int(self.size * fraction)]])

    def getD(self):
        """
            Get dimension.
            \return dimension 
        """
        if self.getSize() == 0: return 0
        return len(self._X[0])

    def getSize(self):
        """
            Get sample size.
            \return sample size 
        """
        return len(self._X)

    def setData(self, X):
        """
            Set data.
            \param X data
        """
        self._X = list(X)

    def setWeights(self, w):
        """
            Set weights.
            \param w weights
        """
        self._W = list(w)

    def isWeighted(self):
        """
            Test if weighted.
            \return True, if the sample has weights.
        """
        return len(self._W) > 0

    def getMean(self, weight=False, fraction=1.0):
        """
            Computes the mean.
            \param weight compute weighted mean
            \param fraction use only upper fraction of the ordered data
        """
        if weight:
            return calc_mean(self.proc_data(fraction=fraction, dtype=float), w=self.proc_weights(fraction=fraction))
        else:
            return calc_mean(self.proc_data(fraction=fraction, dtype=float))

    def getCov(self, weight=False, fraction=1.0):
        """
            Computes the covariance matrix.
            \param weight compute weighted covariance
            \param fraction use only upper fraction of the ordered data
        """
        if weight:
            return calc_cov(X=self.proc_data(fraction=fraction, dtype=float), w=self.proc_weights(fraction=fraction))
        else:
            return calc_cov(X=self.proc_data(fraction=fraction, dtype=float))

    def getCor(self, weight=False, fraction=1.0):
        """
            Computes the correlation matrix.
            \param weight compute weighted correlation
            \param fraction use only upper fraction of the ordered data
        """
        if weight:
            return calc_cor(X=self.proc_data(fraction=fraction, dtype=float), w=self.proc_weights(fraction=fraction))
        else:
            return calc_cor(X=self.proc_data(fraction=fraction, dtype=float))

    def getVar(self, weight=False, fraction=1.0):
        """
            Computes the variance vector.
            \param weight compute weighted variance
            \param fraction use only upper fraction of the ordered data
        """
        return numpy.diag(self.getCov(weight=weight, fraction=fraction))

    def getEss(self):
        """
        Return effective sample size 1/(sum_{w in weights} w^2) .
        """
        if not self.isWeighted(): return 0
        return 1 / (self.size * pow(self.nW, 2).sum())

    def getOrder(self):
        """
            Get order.
            \return index set for the data in ascending order according to the weights.
        """
        if self.__order is None: self.__sort()
        if self.__order is None: self.__order = range(self.size)
        return self.__order

    def __sort(self):
        """
            Sets the index for the data in ascending order according to the weights.
        """
        self.__order = numpy.array(self._W).argsort(axis=0).tolist()
        self.__order.reverse()

    def assign_weights(self, f):
        """
            Evaluates the value of c*exp(f(x)) for each sample x.
            \param f a real-valued function on the sampling space 
        """

        v = self._W
        k = len(v)
        X = numpy.array(self._X)[k:]
        self.nW = []

        # Apply in lexicographical order to avoid extra evaluation of f.
        # lexorder = self.lexorder(X) -- use the numpy function instead
        lexorder = numpy.lexsort(X.T)
        weight = f.lpmf(X[lexorder[0]])
        for index in range(self.size - k):
            if not (X[lexorder[index]] == X[lexorder[index - 1]]).all():
                weight = f.lpmf(X[lexorder[index]])
            self._W.append(weight)
        self._W = numpy.array(self._W)
        self._W = v + list(self._W[numpy.argsort(lexorder)])

    def dichotomize_weights(self, f, fraction):
        w = f.lpmf(numpy.array(self._X))
        order = w.argsort(axis=0).tolist()
        order.reverse()
        k = int(fraction * self.size)
        v = w[order][k]
        self._W = list((w >= v) * 1.0)
        return w[order[0]], self._X[order[0]]

    def distinct(self):
        X = self._X
        W = self.nW

        # order the data array
        lexorder = numpy.lexsort(numpy.array(X).T)

        # check if all entries are equal
        if W[lexorder[0]] == W[lexorder[-1]]:
            self._X = [X[0]]
            self._W = [1.0]
            return
        
        self._X = []; self._W = []
        
        # loop over ordered data
        x, w = X[lexorder[0]], W[lexorder[0]]
        
        count = 1
        for index in numpy.append(lexorder[1:], lexorder[0]):
            if (x == X[index]).all():
                count += 1
            else:
                self._X += [x]
                self._W += [numpy.log(w * count)]
                x = X[index]
                w = W[index]
                count = 1

    def append(self, x, w=None):
        """
            Appends the value x with weigth w to the dataset.
            \param x value
            \param w weight
        """
        if not isinstance(self._X, list): self._X.tolist()
        if not isinstance(self._W, list): self._W.tolist()
        self._X.append(x)
        if not w is None: self._W.append(float(w))

    def shrink(self, index):
        """
            Removes the indicated columns from the data.
            \param index column index set 
        """
        self._X = self.X[:, index]

    def get_sub_data(self, index):
        """
            Returns a shrinked version of the data class.
            \param index column index set 
        """
        data = self.copy()
        data.shrink(index)
        return data

    def copy(self):
        """
            Creates a deep copy.
            \return deep copy
        """
        return copy.deepcopy(self)

    def sample(self, f, size, job_server=None):
        """
            Samples from a random generator.
            \param f random variable
            \param size sample size
            \param verbose print status line
        """
        self._X += list(f.rvs(size, job_server))


    def save(self, filename):
        """
            Saves the sample to a file using pickle.
            \param filename filename
        """
        pickle.dump(self, open(filename, 'wb'))

    def load(self, filename):
        """
            pickle.loads a sample from a file using pickle.
            \param filename filename
        """
        data = pickle.load(open(filename))
        self.__init__(data.X, data._W)

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
    """
        Mean.
        \param X array
        \param w positive weights
        \return mean
    """
    #mean = numpy.average(X, axis=0, weights=numpy.exp(w - w.max()))
    if w is None:
        return X.sum(axis=0) / float(X.shape[0])
    else:
        return (w[:, numpy.newaxis] * X).sum(axis=0)

def calc_cov(X, w=None):
    """
        Covariance.
        \param X array
        \param w positive weights
        \return covariance matrix
    """
    if w is None:
        n = float(X.shape[0])
        mean = calc_mean(X)[numpy.newaxis, :]
        return (numpy.dot(X.T, X) - n * numpy.dot(mean.T, mean)) / float(n - 1)
    else:
        mean = calc_mean(X, w)[numpy.newaxis, :]
        return (numpy.dot(X.T, w[:, numpy.newaxis] * X) - numpy.dot(mean.T, mean)) / (1 - numpy.power(w, 2).sum())

def calc_cor(X, w=None):
    """
        Correlation.
        \param X array
        \param w positive weights
        \return correlation matrix
    """
    d = X.shape[1]
    cov = calc_cov(X, w) + 1e-10 * numpy.eye(d)
    var = cov.diagonal()[numpy.newaxis, :]
    return cov / numpy.sqrt(numpy.dot(var.T, var))

def calc_norm(v, p=2.0):
    return pow(pow(abs(v), p).sum(axis=0), 1.0 / p)
