#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Uniform vectors on a restricted support.
"""

"""
@namespace binary.product_model
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
@details
"""

import binary
import numpy
import utils

class UniformBinary(binary.binary_model.Binary):
    """ Binary model with independent components. """

    name = 'product family'

    def __init__(self, d, q, name='uniform',
                 longname='Binary model with independent components.'):
        """ 
            Constructor.
            \param d dimension
            \param q maximum size
            \param name name
            \param longname longname
        """
        binary.binary_model.Binary.__init__(self, name=name, longname=longname)
        self.name = 'uniform family'
        self.param = dict(d=d, q=q)

        if 'cython' in utils.opts:
            self.f_rvslpmf = utils.cython.uniform_rvslpmf
            self.f_lpmf = utils.cython.uniform_lpmf
            self.f_rvs = utils.cython.uniform_rvs
        else:
            self.f_lpmf = _lpmf
            self.f_rvs = _rvs
            self.f_rvslpmf = _rvslpmf

    def __str__(self):
        return 'maximum size ' + self.param['q']

    @classmethod
    def random(cls, d):
        """ 
            Construct a random product model for testing.
            \param cls class
            \param d dimension
        """
        return cls(d, numpy.random.randint(d) + 1)

    def getP(self):
        p = 0.5 * self.param['q'] / float(self.param['d'])
        return p * numpy.ones(self.param['d'])

    def getD(self):
        """ Get dimension.
            @return dimension 
        """
        return self.param['d']

    p = property(fget=getP, doc="p")


def _lpmf(gamma, param):
    """ 
        Log-probability mass function.
        @param gamma binary vector
        @param param parameters
        @return log-probabilities
    """
    return -param['q'] * numpy.log(2) * numpy.ones(gamma.shape[0])

def _rvs(U, param):
    """ 
        Generates a random variable.
        @param U uniform variables
        @param param parameters
        @return binary variables
    """
    Y = numpy.zeros((U.shape[0], U.shape[1]), dtype=bool)
    d, q = param['d'], param['q']
    for k in xrange(U.shape[0]):
        perm = numpy.arange(d)
        for i in xrange(d):
            # pick an element in p[:i+1] with which to exchange p[i]
            j = int(U[k][i] * (d - i))
            perm[d - 1 - i], perm[j] = perm[j], perm[d - 1 - i]
        # draw the number of nonzero elements
        r = int(U[k][d - 1] * (q + 1))
        Y[k][perm[:r]] = True
    return Y

def _rvslpmf(U, param):
    """ 
        Generates a random variable and computes its probability.
        @param U uniform variables
        @param param parameters
        @return binary variables, log-probabilities
    """
    Y = _rvs(U, param)
    return Y, _lpmf(Y, param)

def main():
    pass

if __name__ == "__main__":
    main()
