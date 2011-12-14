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
import os

import numpy; numpy.seterr(divide='raise')
import scipy.linalg
import binary.base

from binary.logistic_cond import LogisticCondBinary
from binary.linear_cond import LinearCondBinary
from binary.qu_linear import QuLinearBinary
from binary.product import ProductBinary, PositiveProductBinary, ConstrProductBinary, EquableProductBinary, LimitedProductBinary
from binary.gaussian_copula import GaussianCopulaBinary


generator_classes = [
# EquableProductBinary,
# ProductBinary,
# PositiveProductBinary,
# ConstrProductBinary,
# LimitedProductBinary,
# LogisticCondBinary,
# QuLinearBinary,
# GaussianCopulaBinary
# LinearCondBinary
]

n = 1e6
d = 15
phi = 0.5
rho = 0.5
ncpus = 2
rvs_marginals = True
exact_marginals = False

def main():
    compare()

def test_rvs():

    for generator_class in generator_classes:
        generator = generator_class.random(d=d)
        print (generator.name + ' ').ljust(100, '*')
        if rvs_marginals:
            print 'simulation marginals:'
            binary.base.print_moments(generator.rvs_marginals(n, ncpus=ncpus))
        if exact_marginals:
            print 'exact marginals:'
            binary.base.print_moments(generator.exact_marginals(ncpus=ncpus))

def test_properties():

    for generator_class in generator_classes:
        generator_class.test_properties(d, n, phi=phi, ncpus=ncpus)

def compare():

    mean, corr = binary.base.moments2corr(binary.base.random_moments(d, phi=phi, rho=rho))
    #print 'given marginals '.ljust(100, '*')
    #binary.base.print_moments(mean, corr)

    d_corr = corr - numpy.eye(d)
    p_sn, p_fn = scipy.linalg.norm(d_corr, 2), scipy.linalg.norm(d_corr)
    print 'product family:'
    print 'Spectral  norm: %.4f' % p_sn
    print 'Frobenius norm: %.4f\n' % p_fn
    #print repr(d_corr)

    print 'gaussian copula family:'
    generator = GaussianCopulaBinary.from_moments(mean, corr)
    d_corr = corr - generator.corr
    sn, fn = scipy.linalg.norm(d_corr, 2), scipy.linalg.norm(d_corr)
    print 'Spectral  norm: %.4f, %.2f%%' % (sn, (p_sn - sn) / p_sn)
    print 'Frobenius norm: %.4f, %.2f%%\n' % (fn, (p_fn - fn) / p_fn)
    #print repr(d_corr)

    print 'logistic conditionals family:'
    generator = LogisticCondBinary.from_moments(mean, corr)
    d_corr = corr - generator.rvs_marginals(n, ncpus)[1]
    sn, fn = scipy.linalg.norm(d_corr, 2), scipy.linalg.norm(d_corr)
    print 'Spectral  norm: %.4f, %.2f%%' % (sn, (p_sn - sn) / p_sn)
    print 'Frobenius norm: %.4f, %.2f%%\n' % (fn, (p_fn - fn) / p_fn)
    #print repr(d_corr)

    print 'linear conditionals family:'
    generator = LinearCondBinary.from_moments(mean, corr)
    d_corr = corr - generator.rvs_marginals(n, ncpus)[1]
    sn, fn = scipy.linalg.norm(d_corr, 2), scipy.linalg.norm(d_corr)
    print 'Spectral  norm: %.4f, %.2f%%' % (sn, (p_sn - sn) / p_sn)
    print 'Frobenius norm: %.4f, %.2f%%\n' % (fn, (p_fn - fn) / p_fn)
    #print repr(d_corr)

def profile():

    cProfile.run('test_properties()', 'unit_test.prof')
    p = pstats.Stats('unit_test.prof')
    p.sort_stats('time').print_stats(10)
    os.remove('unit_test.prof')

if __name__ == "__main__":
    main()
