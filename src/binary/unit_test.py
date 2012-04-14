#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Run a unit test sampling from all parametric families. \namespace binary.unit_test """

import cProfile
import pstats
import os

import numpy; numpy.seterr(divide='raise')
import binary.base

from binary.conditionals_linear import LinearCondBinary
from binary.conditionals_logistic import LogisticCondBinary
from binary.conditionals_arctan import ArctanCondBinary
from binary.copula_gaussian import GaussianCopulaBinary
from binary.copula_student import StudentCopulaBinary
from binary.product import ProductBinary
from binary.product_exchangeable import ExchangeableBinary
from binary.product_constrained import ConstrProductBinary
from binary.product_limited import LimitedBinary
from binary.product_positive import PositiveBinary
from binary.quadratic_linear import QuLinearBinary

generator_classes = [ExchangeableBinary, ProductBinary, PositiveBinary, ConstrProductBinary, LimitedBinary,
                     StudentCopulaBinary, GaussianCopulaBinary,
                     LogisticCondBinary, LinearCondBinary, ArctanCondBinary,
                     QuLinearBinary]

n = 2e5
d = 4
rho = 0.5
ncpus = 2

def main():
    test_properties()
    
def beep():
    os.system("/usr/bin/canberra-gtk-play --id='bell'")

def test_properties():
    M = binary.base.random_moments(d, rho=rho)
    for generator_class in generator_classes:
        generator_class.test_properties(M=M, n=n, ncpus=ncpus)

def profile():
    cProfile.run('test_properties()', 'unit_test.prof')
    p = pstats.Stats('unit_test.prof')
    p.sort_stats('time').print_stats(10)
    os.remove('unit_test.prof')

if __name__ == "__main__":
    main()
