#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 
    Variable selector for linear normal models.
    @namespace binary.selector_ln
"""

import numpy
import scipy.linalg
import binary.base as base
import os

class SelectorLn(base.BaseBinary):
    """ Variable selector for linear normal models."""

    name = 'selector ln'

    def __init__(self, y, Z, config, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param Y explained variable
            \param Z covariates to perform selection on
            \param parameter dictionary
        """

        self.static = config['data/static']

        super(SelectorLn, self).__init__(d=Z.shape[1] - self.static, name=name, long_name=long_name)

        # add modules
        self.pp_modules += ('scipy.linalg', 'binary.selector_ln',)

        # normalize
        self.Z = numpy.subtract(Z, Z.mean(axis=0))
        self.y = y
        self.Zty = numpy.dot(self.Z.T, y)

        # store parameters
        self.n = Z.shape[0]
        self.TSS = None

        # prior on marginal inclusion probability
        p = config['prior/model_inclprob']
        if p is None: p = 0.5
        self.LOGIT_P = numpy.log(p / (1.0 - p))

        # maximum model size
        self.max_size = config['prior/model_maxsize']
        if self.max_size is None:
            self.max_size = numpy.inf
        elif isinstance(self.max_size, str):
            if self.max_size == 'n':
                self.max_size = self.n
        self.max_size

        self.constraints = config['data/constraints']
        self.PENALTY = -self.n * self.tss


    def __str__(self):

        ZtZ = numpy.dot(self.Z.T, self.Z) + 1e-10 * numpy.eye(self.dstar)

        # null model
        sigma2_null = self.tss / float(self.n - 2)

        # only static components
        if self.static > 0:
            i = numpy.zeros(self.dstar, dtype=bool)
            i[:self.static] = True
            sigma2_fixed = (self.tss - numpy.dot(self.Zty[i, :], scipy.linalg.solve(ZtZ[i, :][:, i], self.Zty[i, :], sym_pos=True)))
            sigma2_fixed /= float(self.n - 2)
        else:
            sigma2_fixed = sigma2_null

        # full model
        if self.n > self.dstar:
            sigma2_full = (self.tss - numpy.dot(self.Zty, scipy.linalg.solve(ZtZ, self.Zty, sym_pos=True)))
            sigma2_full /= float(self.n - 2)
        else:
            sigma2_full = 0.0

        template = """Problem summary:
                    > sigma^2_null     : %f
                    > sigma^2_fixed    : %f
                    > sigma^2_full     : %f
                    > number of obs    : %d
                    > number of covs   : %d
                    > number of pcs    : %d""".replace(20 * ' ', '')

        return template % (sigma2_null, sigma2_fixed, sigma2_full, self.n, self.dstar, self.static)

    def get_tss(self):
        """ Total sum of squares.
            @return total sum of squares
        """
        if self.TSS is None: self.TSS = numpy.dot(self.y.T, self.y) - (self.y.sum() ** 2) / float(self.y.shape[0])
        return self.TSS

    def get_dstar(self):
        """ Get dimension of the extended problem.
            @return dimension 
        """
        return self.d + self.static

    dstar = property(fget=get_dstar, doc="dimension of the extended problem")
    tss = property(fget=get_tss, doc="total sum of squares")

    @classmethod
    def _lpmf(cls, Y, config):
        """ 
            Log-posterior probability mass function in a hierarchical Bayesian model.
            
            \param Y binary vector
            \param param parameters
            \return log-probabilities
        """

        # array to store results
        L = numpy.empty(Y.shape[0], dtype=float)
        Ystar = numpy.ones(shape=config.dstar, dtype=bool)

        for k in xrange(Y.shape[0]):

            # model dimension
            Ystar[config.static:] = Y[k]
            size = Ystar.sum()

            # check main effects constraints
            if len(config.constraints) > 0:
                violations = (Ystar[config.constraints[:, 0]] >
                                  Ystar[config.constraints[:, 1]] * Ystar[config.constraints[:, 2]]).sum()
            else:
                violations = False

            if violations > 0 or size > config.max_size:
                # inadmissible model
                L[k] = config.PENALTY - violations - size
            else:
                # regular model
                L[k] = cls.score(Ystar, config, size)

        return L

def main():

    path = os.path.expanduser('~/Documents/Data/bvs/test')
    n = 200
    p = 10
    d = 100

    beta = numpy.random.standard_normal(size=p)
    X = numpy.random.standard_normal(size=(n, d))
    y = numpy.dot(X[:, :p], beta)

    f = open(os.path.join(path, 'test.csv'), 'w')
    f.write(','.join(['y'] + ['x%d' % (i + 1) for i in xrange(d)]) + '\n')
    for k in xrange(n):
        f.write(','.join(['%.6f' % y[k]] + ['%.6f' % x for x in X[k]]) + '\n')
    f.close()

if __name__ == "__main__":
    main()
