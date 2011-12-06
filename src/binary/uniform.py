#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family uniform on certain subsets. """

"""
@namespace binary.uniform
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

import numpy

import binary.base
import binary.wrapper

def _lpmf(gamma, param):
    """ 
        Log-probability mass function.
        \param gamma binary vector
        \param param parameters
        \return log-probabilities
    """
    return -param['q'] * numpy.log(2) * numpy.ones(gamma.shape[0])

def _rvs(U, param):
    """ 
        Generates a random variable.
        \param U uniform variables
        \param param parameters
        \return binary variables
    """
    #if param['hasCython']: return binary.uniform_ext._rvs(q=param['q'], U=U)

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
        \param U uniform variables
        \param param parameters
        \return binary variables, log-probabilities
    """
    Y = _rvs(U, param)
    return Y, _lpmf(Y, param)


class UniformBinary(binary.base.BaseBinary):
    """ Binary parametric family uniform on certain subsets. """

    def __init__(self, d, q, py_wrapper=None, name='uniform family', long_name=__doc__):
        """ 
            Constructor.
            \param d dimension
            \param q maximum size
            \param name name
            \param long_name long_name
        """

        if py_wrapper is None: py_wrapper = binary.wrapper.uniform()

        binary.base.BaseBinary.__init__(self, py_wrapper=py_wrapper, name=name, long_name=long_name)

        self.param.update({'d':d, 'q':q})
        self.pp_modules = ('numpy', 'binary.uniform')


    def __str__(self):
        return 'maximum size %d' % self.param['q']

    @classmethod
    def random(cls, d):
        """ 
            Construct a random product model for testing.
            \param cls class
            \param d dimension
        """
        return cls(d, numpy.random.randint(d) + 1)

    def _getMean(self):
        mean = 0.5 * self.param['q'] / float(self.param['d'])
        return mean * numpy.ones(self.param['d'])

    def _getD(self):
        """ Get dimension of instance. \return dimension """
        return self.param['d']

def main():
    n, d, max_size = 500, 40, 10
    u = UniformBinary(d, max_size)
    X = u.rvs(n)
    print X.sum(axis=0) / float(n)
    print u.mean

if __name__ == "__main__":
    main()
