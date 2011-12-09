#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Run a unit test sampling from all parametric families."""

"""
\namespace binary.unittest
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

import cProfile
import pstats

import numpy

import binary.base

from binary.logistic_cond import LogisticCondBinary
from binary.qu_linear import QuLinearBinary
from binary.product import ProductBinary, PositiveProductBinary, \
    ConstrProductBinary, EquableProductBinary, LimitedProductBinary

generator_classes = [
 EquableProductBinary,
 ProductBinary,
 PositiveProductBinary,
 ConstrProductBinary,
 LimitedProductBinary,
 LogisticCondBinary,
 QuLinearBinary,
]

n = 1500
d = 15
phi = 1
ncpus = 1
rvs_marginals = True
exact_marginals = True

def test_rvs():

    for generator_class in generator_classes:
        generator = generator_class.random(d=d)
        print '\n' + 50 * '*' + '\n' + generator.name
        print generator
        if rvs_marginals:
            print "sample (n = %(n)d) mean:\n%%s\nsample (n = %(n)d) correlation:\n%%s" % {'n':n} % generator.rvs_marginals(n, ncpus=ncpus)
        if exact_marginals:
            print "exact mean:\n%s\nexact correlation:\n%s" % generator.exact_marginals(ncpus=ncpus)

def test_qu_linear():

    #print 'given '.ljust(100, '*')
    #print_marginals(mean, corr)

    #mean, corr = binary.base.moments2corr(binary.base.random_moments(d, phi=phi))
    #generator = QuLinearBinary.from_moments(mean, corr)
    #generator = QuLinearBinary.random(d)
    generator = QuLinearBinary(a=0, Beta=random_ubqo_problem(d))

    print generator.name + ':'
    print generator

    print 'formula '.ljust(100, '*')
    print_marginals(generator.mean, generator.cov)

    X = generator.state_space()

    print 'exact (by summation)'.ljust(100, '*')
    weights = generator.pmf(X)
    print 'max', weights.max()
    print_marginals(generator.exact_marginals(ncpus))
    print X[weights.argmax()]

    print 'exact (by conditionals) '.ljust(100, '*')
    weights = generator.pmf_reject(X)
    print 'max', weights.max()
    if weights.sum() == 0:
        print 'Degenerated probability law. Stopped.'
    else:
        weights /= weights.sum()
        mean, corr = binary.base.sample2corr(X, weights)
        print_marginals(mean, corr)
    
        generator = QuLinearBinary.from_moments(mean, corr)
        print generator
        weights = generator.pmf(X)
        print 'max', weights.max()
        print X[weights.argmax()]

    #print ('sample (n = %d) ' % n).ljust(100, '*')
    #mean, corr = generator.rvs_marginals(n, ncpus=ncpus)
    #print_marginals(mean, corr)


def test_logistic_cond():

    mean, corr = binary.base.moments2corr(binary.base.random_moments(d, phi))
    print 'given mean:\n' + repr(mean)
    print 'given correlation:\n' + repr(corr)

    generator = LogisticCondBinary.from_moments(mean, corr)
    print generator.exact_marginals()

def print_marginals(mean, corr=None):
    if isinstance(mean, tuple):
        mean, corr = mean
    print 'mean:\n' + repr(mean)
    print 'correlation:\n' + repr(corr)

def random_ubqo_problem(d, rho=1.0, c=50, generator=numpy.random.standard_normal):
    """ Generates a random UBQO problem. \return symmetric matrix"""
    A = numpy.zeros((d, d))
    for i in xrange(d):
        A[i, :i + 1] = generator(i + 1).T * c // 1 * (numpy.random.random(size=i + 1) <= rho)
        A[:i, i] = A[i, :i]
        if A[i, i] == 0: A[i, i] = 1
    return A

if __name__ == "__main__":

    generator_classes = [QuLinearBinary]

    test_qu_linear()

    #cProfile.run('rvstest()', 'unit_test.prof')
    #p = pstats.Stats('unit_test.prof')
    #p.sort_stats('time').print_stats(10)

