#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Run a unit test sampling from all parametric families. \namespace binary.unit_test """

import cProfile
import pstats
import os

import numpy; numpy.seterr(divide='raise')
import scipy.linalg
import binary.base

from binary import *
from logistic_cond import LogisticCondBinary
from linear_cond import LinearCondBinary
from qu_linear import QuLinearBinary
from product import ProductBinary
from positive import PositiveBinary
from constrained import ConstrProductBinary
from exchangeable import ExchangeableBinary
from limited import LimitedBinary
from conditionals import ConditionalsBinary
from gaussian_copula import GaussianCopulaBinary
from student_copula import StudentCopulaBinary

generator_classes = [ConditionalsBinary]

n = 2e5
d = 4
rho = 1.0
ncpus = 2
rvs_marginals = True
exact_marginals = False

def main():
    #test_properties()
    #exit(0)
    M = binary.base.random_moments(d, rho, n_cond=100, n_perm=10)
    mean1, corr1 = binary.base.moments2corr(M)
    print repr(M)

    generator = StudentCopulaBinary.from_moments(mean1, corr1, delta=0)
    mean, corr = generator.mean, generator.corr
    print generator.name
    print repr(binary.base.corr2moments(mean, corr))

    generator = GaussianCopulaBinary.from_moments(mean1, corr1, delta=0)
    mean, corr = generator.mean, generator.corr #generator.rvs_marginals(n, ncpus=ncpus) #
    print generator.name
    print repr(binary.base.corr2moments(mean, corr))

    '''
    generator = GaussianCopulaBinary.from_moments(mean, corr, delta=0)
    mean, corr = generator.mean, generator.corr
    print 'Gaussian'
    print repr(binary.base.corr2moments(mean, corr))

    generator = LogisticCondBinary.from_moments(mean, corr, verbose=0, delta=0)
    mean, corr = generator.exact_marginals(ncpus=ncpus)
    print 'Logistic'
    print repr(binary.base.corr2moments(mean, corr))
    print generator
    '''


def beep():
    os.system("/usr/bin/canberra-gtk-play --id='bell'")

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
        generator_class.test_properties(d, n, rho=rho, ncpus=ncpus)

def profile():

    cProfile.run('test_properties()', 'unit_test.prof')
    p = pstats.Stats('unit_test.prof')
    p.sort_stats('time').print_stats(10)
    os.remove('unit_test.prof')


if __name__ == "__main__":
    main()
