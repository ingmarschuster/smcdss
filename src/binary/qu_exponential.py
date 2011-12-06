#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with exponential quadratic form. """

"""
\namespace binary.qu_exponential
$Author: christian.a.schafer@gmail.com $
$Rev: 149 $
$Date: 2011-09-16 17:40:26 +0200 (Fr, 16 Sep 2011) $
"""

import numpy
import product
import scipy.linalg
import scipy.stats
import logistic_cond
import utils

def _lpmf(gamma, param):
    """ 
        Log-probability mass function.
        \param gamma binary vector
        \param param parameters
        \return log-probabilities
    """
    L = numpy.empty(gamma.shape[0])
    for k in xrange(gamma.shape[0]):
        L[k] = float(numpy.dot(numpy.dot(gamma[k], param['A']), gamma[k].T))
    return L

class QuExpBinary(product.ProductBinary):
    """ Binary parametric family with quadratic exponential term. """

    def __init__(self, A, name='quadratic exponential binary', long_name=__doc__):
        """ 
            Constructor.
            \param A matrix of coefficients
        """

        product.ProductBinary.__init__(self, name=name, long_name=long_name)
        self.f_lpmf = _lpmf
        self.f_rvs = None
        self.f_rvslpmf = None
        self.param = dict(A=A)

    @classmethod
    def independent(cls, p):
        """
            Constructs a quadratic exponential family with independent components.
            \param cls class
            \param p mean
        """
        return cls(numpy.diag(numpy.log(p / (1 - p))))

    @classmethod
    def random(cls, d, scale=0.5):
        """
            Constructs a random quadratic exponential family for testing.
            \param cls class 
            \param d dimension
            \param scale standard deviation of the off-diagonal elements
        """
        p = numpy.random.random(d)
        logratio = numpy.log(p / (1 - p))
        A = numpy.diag(logratio)
        for i in xrange(d):
            if scale > 0.0: A[i, :i] = numpy.random.normal(scale=scale, size=i)

        return QuExpBinary(A)

    def __str__(self):
        return utils.format.format_matrix(self.A, 'A')

    def _getD(self):
        """ Get dimension of instance. \return dimension """
        return self.param['A'].shape[0]

    def getA(self):
        return self.param['A']

    A = property(fget=getA, doc="A")


def calc_marginal(A):
    """
        Computes a quadratic exponential model where the component that causes
        the least quadratic approximation error has been marginalized. The
        method is inspired by Cox and Wermuth, A note on the quadratic
        exponential binary distribution, Biometrika 1994, 81, 2, pp. 403-8.
        
        \param A coefficient matrix
        \return model the approximate marginal distribution
    """

    I = range(A.shape[0])
    d = len(I)

    num = 10
    w = numpy.empty(num)
    p = 0.5
    for j in xrange(num):
        w[j] = scipy.stats.binom.pmf(j, num - 1, p)
    w = numpy.sqrt(w)

    X = numpy.ones((num, 3))
    Beta = numpy.zeros((d, d))
    perm = list()

    while d > 0:
        l, u, B = numpy.empty(d), numpy.empty(d), numpy.empty((d, d - 1))

        # determine range of each component
        for k in range(d):
            B[k] = numpy.concatenate((A[k, :k], A[k + 1:, k]))
            l[k] = max(B[k][B[k] < 0].sum() + A[k, k], -15)
            u[k] = min(B[k][B[k] > 0].sum() + A[k, k], 15)

        # compute least square regression and error
        xi, eps = list(), list()
        for k in range(d):
            if u[k] - l[k] > 0:
                x = numpy.linspace(l[k], u[k], num=num)
                y = w * numpy.log(numpy.cosh(x))
                X[:, 1] = x
                X[:, 2] = numpy.power(x, 2)
                W = w[:, numpy.newaxis] * X
                xi.append(scipy.linalg.solve(numpy.dot(W.T, W), numpy.dot(W.T, y)))
                eps.append(numpy.linalg.norm(numpy.dot(W, xi[-1]) - y))
            else:
                xi.append(numpy.zeros(3))
                eps.append(0)

        # pick component with least error
        k = eps.index(min(eps))
        xi = xi[k]

        # the original Taylor series approach corresponds to 
        # xi = [0, 0.5 * numpy.tanh(0.5 * A[k, k]), 0.125 / (numpy.cosh(0.5 * A[k, k])** 2)]

        r = range(k) + range(k + 1, d)
        b = numpy.concatenate((B[k], numpy.array([A[k, k]])))
        A = A[r, :][:, r] + xi[1] * numpy.diag(B[k]) + (0.5 + xi[2]) * numpy.tril(numpy.outer(B[k], B[k]))

        perm.append(I[k])
        Beta[d - 1, :d] = b

        I.pop(k)
        d = len(I)

    model = logistic_cond.LogisticCondBinary(Beta)
    perm.reverse()
    model.v2m_perm = perm
    return model

def calc_logistic_model(A):
    """
        Constructs a logistic conditional model from a quadratic exponential model by
        repeatedly computing the approximate marginal distributions
        \param A coefficient matrix
        \return model a logistic conditional model
    """
    d = A.shape[0]
    Beta = numpy.zeros((d, d))
    perm = list()

    for i in xrange(d - 1, -1, -1):
        A, b, k = calc_marginal(A)
        Beta[i, :i + 1] = b
        perm.append(k)

    model = logistic_cond.LogisticCondBinary(Beta)
    model.m2v_perm = perm
    return model

def main():
    pass

if __name__ == "__main__":
    main()
