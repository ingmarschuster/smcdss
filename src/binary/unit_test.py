#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Run a unit test sampling from all parametric families."""

"""
\namespace binary.unit_test
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

import cProfile
import pstats
import os, sys

import numpy; numpy.seterr(divide='raise')
import binary.base
import scipy.linalg

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
LogisticCondBinary,
# QuLinearBinary,
# GaussianCopulaBinary
# LinearCondBinary
]

n = 1e5
d = 8
phi = 1.0
rho = 1.0
ncpus = 1
rvs_marginals = True
exact_marginals = False

def main():

    '''
    M = random_moments(d)
    print repr(M)
    mean, corr = binary.base.moments2corr(M)
    print repr(corr)
    try:
        scipy.linalg.cholesky(corr)
        print 'positive definite.'
    except scipy.linalg.LinAlgError:
        print 'NOT positive definite!'
    exit(0)
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
    N = 10
    n = 2
    m = float(n) / float(N) * numpy.ones(N)

    M = numpy.outer(m, m) + numpy.diag(m - pow(m, 2))
    cov = M - numpy.outer(m, m)
    print cov.sum()
    c = 0.7455
    M = numpy.outer(c * m, c * m) + numpy.diag(m - pow(c * m, 2))
    cov = M - numpy.outer(m, m)
    print cov
    print cov.sum()
    try:
        scipy.linalg.cholesky(cov)
        print 'positive definite.'
    except scipy.linalg.LinAlgError:
        print 'NOT positive definite!'
        exit(0)
    mean, corr = binary.base.moments2corr(M)
    generator = LogisticCondBinary.from_moments(mean, corr, verbose=0, delta=0)
    print generator
    for i in xrange(50):
        gamma = generator.rvs()
        print gamma.sum()

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
        generator_class.test_properties(d, n, phi=phi, ncpus=ncpus)

def profile():

    cProfile.run('test_properties()', 'unit_test.prof')
    p = pstats.Stats('unit_test.prof')
    p.sort_stats('time').print_stats(10)
    os.remove('unit_test.prof')

def random_moments(d, rho=1.0, verbose=False):
    """
        Creates a random cross-moments matrix that is consistent with the
        general constraints on binary data.
        \param d dimension
        \param eps minmum distance to borders of [0,1]
        \param phi parameter in [0,1] where phi=0 means zero correlation
        \return M cross-moment matrix
    """
    N_CONDITIONALS = int(5e2)
    N_PERMUTATIONS = int(10 * d)
    BOUNDARY = 0.01

    # mean
    m = BOUNDARY + numpy.random.random(d) * (1.0 - 2 * BOUNDARY)

    # covariance of independent Bernoulli
    C = numpy.diag(m - m * m)

    i = d - 1
    b_lower, b_upper = numpy.empty(i), numpy.empty(i)

    for l in xrange(N_PERMUTATIONS):

        if l % 1e1 == 0 and verbose:
            sys.stderr.write('perm: %.0f\n' % (l / 1e1))

        # compute bounds
        for j in xrange(i):
            b_lower[j] = max(m[i] + m[j] - 1.0, 0)
            b_upper[j] = min(m[i], m[j])
            for bound in [b_lower, b_upper]: bound[j] -= m[i] * m[j]

        # compute inverse
        C_inv = scipy.linalg.inv(C[:i, :i])
        det = C[i, i] - numpy.dot(numpy.dot(C[:i, i], C_inv), C[:i, i])

        for k in xrange(N_CONDITIONALS):

            for j in getPermutation(i):

                det_j = numpy.dot(C[:i, i], C_inv[:, j]) - C_inv[j, j] * C[i, j]
                rest = det + (C_inv[j, j] * C[i, j] + 2 * det_j) * C[i, j]

                # maximal feasible range to ensure finiteness
                t = det_j / C_inv[j, j]
                root = numpy.sqrt(t * t + rest / C_inv[j, j])

                # compute bounds
                lower = max(-t - root, b_lower[j])
                upper = min(-t + root, b_upper[j])

                # avoid extreme cases
                adjustment = (1.0 - rho) * 0.5 * (upper - lower)
                lower += adjustment
                upper -= adjustment

                # draw conditional entry
                C[i, j] = lower + (upper - lower) * numpy.random.random()
                C[j, i] = C[i, j]

                det_j = numpy.dot(C[:i, i], C_inv[:, j])
                det = rest - (2 * det_j - C_inv[j, j] * C[i, j]) * C[i, j]

        # draw random permutation
        perm = getPermutation(d)
        C = C[perm, :][:, perm]
        m = m[perm]

    return C + numpy.outer(m, m)

def getPermutation(d):
    """
        Draw a random permutation of the index set.
        \return permutation
    """
    perm = range(d)
    for i in reversed(range(1, d)):
        # pick an element in p[:i+1] with which to exchange p[i]
        j = numpy.random.randint(low=0, high=i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return numpy.array(perm)

if __name__ == "__main__":
    main()
