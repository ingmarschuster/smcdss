#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 
    Posterior distribution for Bayesian variable selection.
    @namespace binary.posterior
    @details Reads a dataset and construct the posterior probabilities of all linear models
             with variables regressed on the first column.
"""

import numpy
import scipy.linalg
import binary.wrapper as wrapper
import binary.base as base
import os

class PosteriorLink(base.BaseBinary):
    """ Posterior distribution for Bayesian variable selection."""

    name = 'posterior'

    def __init__(self, y, Z, config, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param Y explained variable
            \param Z covariates to perform selection on
            \param parameter dictionary
        """

        self.static = config['data/static']

        super(PosteriorLink, self).__init__(d=Z.shape[1] - self.static, name=name, long_name=long_name)

        # add modules
        self.py_wrapper = wrapper.posterior_link()
        self.pp_modules += ('binary.posterior_link',)

        # normalize
        self.Z = numpy.subtract(Z, Z.mean(axis=0))
        self.y = y
        self.Zty = numpy.dot(self.Z.T, y)

        # store parameters
        self.n = Z.shape[0]

    @classmethod
    def _lpmf(cls, Y, config):
        """ 
            Log-posterior probability mass function.
            
            \param Y binary vector
            \param param parameters
            \return log-probabilities
        """

        # array to store results
        L = numpy.empty(Y.shape[0], dtype=float)
        Y = numpy.ones(shape=config.dstar, dtype=bool)

        for k in xrange(Y.shape[0]):
            L[k] = Y[k].sum()

        return L

    @classmethod
    def link(cls, x):
        return 1.0 / (1.0 + numpy.exp(-x))

def main():

    path = os.path.expanduser('~/Documents/Data/bvs/test')
    n = 200
    p = 10
    d = 20

    beta = numpy.random.standard_normal(size=p)
    X = numpy.random.standard_normal(size=(n, d))
    y = PosteriorLink.link(numpy.dot(X[:, :p], beta))

    f = open(os.path.join(path, 'test_link.csv'), 'w')
    f.write(','.join(['y'] + ['x%d' % (i + 1) for i in xrange(d)]) + '\n')
    for k in xrange(n):
        f.write(','.join(['%.6f' % y[k]] + ['%.6f' % x for x in X[k]]) + '\n')
    f.close()

if __name__ == "__main__":
    main()
