#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Profile of parametric families.
@namespace binary.compare.profiler
"""

from __init__ import *
import binary.base
import cProfile
import os
import pstats

generator_classes = [
ExchangeableBinary,
ProductBinary,
PositiveBinary,
ConstrProductBinary,
LimitedBinary,
StudentCopulaBinary,
GaussianCopulaBinary,
LogisticCondBinary,
LinearCondBinary,
ArctanCondBinary,
QuLinearBinary]

n = 2e3
d = 4
rho = 1.0
ncpus = 2

def main():
    test_properties()

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
