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

from binary.base import print_moments

from binary.logistic_cond import LogisticCondBinary
from binary.qu_linear import QuLinearBinary
from binary.product import ProductBinary, PositiveProductBinary, ConstrProductBinary, EquableProductBinary, LimitedProductBinary
from binary.gaussian_copula import GaussianCopulaBinary


generator_classes = [
# EquableProductBinary,
# ProductBinary,
# PositiveProductBinary,
# ConstrProductBinary,
# LimitedProductBinary,
 LogisticCondBinary,
 QuLinearBinary,
 GaussianCopulaBinary
]

n = 50000
d = 5
phi = 1.0
ncpus = 1
rvs_marginals = True
exact_marginals = False

def main():
    test_properties()

def test_rvs():

    for generator_class in generator_classes:
        generator = generator_class.random(d=d)
        print (generator.name + ' ').ljust(100, '*')
        if rvs_marginals:
            print 'simulation marginals:'
            print_moments(generator.rvs_marginals(n, ncpus=ncpus))
        if exact_marginals:
            print 'exact marginals:'
            print_moments(generator.exact_marginals(ncpus=ncpus))

def test_properties():

    for generator_class in generator_classes:
        generator_class.test_properties(d, n, phi=phi, ncpus=ncpus)

def profile():

    cProfile.run('test_rvs()', 'unit_test.prof')
    p = pstats.Stats('unit_test.prof')
    p.sort_stats('time').print_stats(10)
    os.remove('unit_test.prof')

if __name__ == "__main__":
    main()
