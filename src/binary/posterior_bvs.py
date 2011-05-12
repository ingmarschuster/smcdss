#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Posterior distribution of a model selection problem.
"""

"""
@namespace binary.posterior_bvs
$Author$
$Rev$
$Date$
@details Reads a dataset and construct the posterior probabilities of all linear models
with variables regressed on the first column.
"""

from binary import *

class PosteriorBinary(binary_model.Binary):
    """
        Reads a dataset and construct the posterior probabilities of all linear
        models with variables regressed on the first column.
    """
    def __init__(self, Y, X, param):
        """ 
            Constructor.
            @param Y explained variable
            @param X covariate to perform selection on
            @param param parameter dictonary
        """

        binary_model.Binary.__init__(self, name='posterior-binary', longname='A posterior distribution of a Bayesian variable selection problem.')

        n = X.shape[0]
        d = X.shape[1]
        XtY = numpy.dot(X.T, Y)
        XtX = numpy.dot(X.T, X)
        YtY = numpy.dot(Y.T, Y)

        self.param = dict(XtX=XtX, XtY=XtY, n=n, d=d)

        ## Hierachical Bayesian (hb) or Bayesian Information Criterion (bic)
        if param['POSTERIOR_TYPE'] == 'hb': self._init_hb(n, d, XtY, XtX, YtY, param)
        if param['POSTERIOR_TYPE'] == 'bic': self._init_bic(n, d, XtY, XtX, YtY, param)

    def _init_hb(self, n, d, XtY, XtX, YtY, param):
        """ 
            Setup Hierachical Bayesian posterior.
            @param n sample size
            @param d dimension
            @param XtY X times Y
            @param XtX X times X
            @param YtY T times Y            
            @param param parameter dictonary
        """
        self.f_lpmf = _lpmf_hb

        # full linear model
        beta_full_LM = scipy.linalg.solve(XtX + 1e-5 * numpy.eye(d), XtY, sym_pos=True)
        sigma2_full_LM = (YtY - numpy.dot(XtY, beta_full_LM)) / float(n)

        # prior (normal) of beta
        u = param['PRIOR_BETA_PARAM_U2']
        if u is None: u = 10.0
        v = u / (sigma2_full_LM + 1e-5)

        # prior (inverse gamma) of sigma^2
        lambda_ = param['PRIOR_SIGMA_PARAM_LAMBDA']
        if lambda_ is None: lambda_ = sigma2_full_LM

        # choose nu such that the prior mean is 2 * sigma2_full_LM
        w = param['PRIOR_SIGMA_PARAM_W']

        # prior (bernoulli) of gamma
        p = param['PRIOR_GAMMA_PARAM_P']

        # costants
        c1, c2, c3 = 0.5 * numpy.log(v), 0.5 * (n + w), w * lambda_ + YtY

        self.param.update(dict(v=v, c1=c1, c2=c2, c3=c3, logit_p=utils.logit(p)))


    def _init_bic(self, n, d, XtY, XtX, YtY, param):
        """ 
            Setup Hierachical Bayesian posterior.
            @param n sample size
            @param d dimension
            @param XtY X times Y
            @param XtX X times X
            @param YtY T times Y            
            @param param parameter dictonary
        """
        self.f_lpmf = _lpmf_bic

        # costants
        c1, c2, c3 = YtY / float(n), XtY / float(n), 0.5 * numpy.log(n)

        self.param.update(dict(c1=c1, c2=c2, c3=c3))


    def getD(self):
        """ Get dimension.
            @return dimension 
        """
        return self.param['XtX'].shape[0]

    def __explore(self):
        """ Find the maximmum of the log-probability.
            @deprecated method is never used.
        """
        ## level of logarithm
        self.loglevel = -numpy.inf
        for dec in range(2 ** self.d):
            bin = utils.format.dec2bin(dec, self.d)
            eval = self.lpmf(bin)
            if eval > self.loglevel: self.loglevel = eval

    def pmf(self, gamma):
        """ Unnormalized probability mass function.
            @param gamma binary vector
        """
        if not hasattr(self, 'loglevel'): self.__explore()
        return numpy.exp(self.lpmf(gamma) - self.loglevel)


def _lpmf_hb(gamma, param):
    """ 
        Log-posterior probability mass function in a Hierarchical Bayes model.
        @param gamma binary vector
        @param param parameters
        @return log-probabilities
    """

    # unpack parameters
    XtY, XtX, c1, c2, c3, logit_p = [param[k] for k in ['XtY', 'XtX', 'c1', 'c2', 'c3', 'logit_p']]

    # number of models
    size = gamma.shape[0]
    # array to store results
    L = numpy.empty(size, dtype=float)

    for k in xrange(size):
        # model dimension
        d = gamma[k].sum()
        if d == 0:
            # degenerated model
            L[k] = -numpy.inf
        else:
            # regular model
            C = scipy.linalg.cholesky(XtX[gamma[k], :][:, gamma[k]] + param['v'] * numpy.eye(d))
            if C.shape == (1, 1): b = XtY[gamma[k], :] / float(C)
            else:                 b = scipy.linalg.solve(C.T, XtY[gamma[k], :])
            log_diag_C = numpy.log(C.diagonal()).sum()
            L[k] = -log_diag_C - c1 * d - c2 * numpy.log(c3 - numpy.dot(b, b.T)) + d * logit_p

    return L


def _lpmf_bic(gamma, param):
    """ 
        Score of the Bayesian Information Criterion.
        @param gamma binary vector
        @param param parameters
        @return score
    """

    # unpack parameters
    XtY, XtX, c1, c2, c3, n = [param[k] for k in ['XtY', 'XtX', 'c1', 'c2', 'c3', 'n']]

    # number of models
    size = gamma.shape[0]
    # array to store results
    L = numpy.empty(size, dtype=float)

    for k in xrange(size):
        # model dimension
        d = gamma[k].sum()
        if d == 0:
            # degenerated model
            L[k] = -numpy.inf
        else:
            # regular model
            try:
                beta = scipy.linalg.solve(XtX[gamma[k], :][:, gamma[k]], XtY[gamma[k], :], sym_pos=True)
            except scipy.linalg.LinAlgError:
                # improve condition
                beta = scipy.linalg.solve(XtX[gamma[k], :][:, gamma[k]] + 1e-5 * numpy.eye(d), XtY[gamma[k], :], sym_pos=True)
            L[k] = -0.5 * n * numpy.log(c1 - numpy.dot(c2[gamma[k]], beta)) - c3 * d

    return L
