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
import sys

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

        # We copy some parts of the dictionary since the complete one cannot be
        # pickled which is necessary for parallel python to work properly
        self.param = {'penalty':-n * numpy.log(YtY), 'XtX':XtX, 'XtY':XtY, 'n':n, 'd':d, 'DATA_PCA':param['DATA_PCA'], 'INTERACTIONS':param['INTERACTIONS']}

        self.type = param['POSTERIOR_TYPE']
        
        ## Hierachical Bayesian (hb), Bayesian Information Criterion (bic) or Random Effect (re)
        if self.type == 'hb': self._init_hb(n, d, XtY, XtX, YtY, X, Y, param)
        if self.type == 'bic': self._init_bic(n, d, XtY, XtX, YtY, param)
        if self.type == 're': self._init_re(n, X, Y, param)

    def _init_re(self, n, X, Y, param):
        """ 
            Setup Random Effect model.
            @param n sample size
            @param d dimension
            @param XtY X times Y
            @param XtX X times X
            @param YtY T times Y            
            @param param parameter dictonary
        """
        self.f_lpmf = _lpmf_re
        d = len(param['GROUPS'])

        # Compute kinship matrices
        sample_pos, groups = 0, list()
        for i in xrange(d):
            gd = param['GROUPS'][i]['end'] - param['GROUPS'][i]['start'] + 1
            X_pos = X[:, range(sample_pos, sample_pos + gd)].T
            X_neg = numpy.ones(shape=(gd, n)) - X_pos
            groups += [(numpy.dot(X_pos.T, X_pos) + numpy.dot(X_neg.T, X_neg)) / float(gd)]
            sample_pos += gd
        # normalize Y
        Y -= numpy.ones(n) * (Y.sum() / float(n))
        self.param.update({'GROUPS':groups, 'd':d, 'Y':Y})


    def _init_hb(self, n, d, XtY, XtX, YtY, X, Y, param):
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
        beta_full_LM = scipy.linalg.solve(XtX + 1e-10 * numpy.eye(d), XtY, sym_pos=True)
        sigma2_full_LM = (YtY - numpy.dot(XtY, beta_full_LM)) / float(n - 2)

        # pca model
        if param['DATA_PCA'] > 0:
            gamma_pca = numpy.zeros(d, dtype=bool)
            gamma_pca[:param['DATA_PCA']] = True
            beta_pca_LM = scipy.linalg.solve(XtX[gamma_pca, :][:, gamma_pca] + 1e-10 * numpy.eye(param['DATA_PCA']), XtY[gamma_pca, :], sym_pos=True)
            sigma2_pca_LM = (YtY - numpy.dot(XtY[gamma_pca, :], beta_pca_LM)) / float(n - 2)
        else:
            sigma2_pca_LM = sigma2_full_LM

        sigma2_const_LM = (numpy.power(Y - Y.sum() / float(n), 2).sum() / float(n - 2))

        # prior (normal) of beta
        u2 = param['PRIOR_BETA_PARAM_U2']
        if u2 is None: u2 = 10.0
        v2 = u2 / (sigma2_full_LM + 1e-5)

        # prior (inverse gamma) of sigma^2
        lambda_ = param['PRIOR_SIGMA_PARAM_LAMBDA']
        if lambda_ is None: lambda_ = sigma2_full_LM

        # choose nu such that the prior mean is 2 * sigma2_full_LM
        w = param['PRIOR_SIGMA_PARAM_W']

        # prior (bernoulli) of gamma
        p = param['PRIOR_GAMMA_PARAM_P']

        # costants
        self.param.update({'one_over_v2':1.0 / v2,
                           'c1' : 0.5 * numpy.log(v2),
                           'c2' : 0.5 * (n + w),
                           'c3' : w * lambda_ + YtY,
                           'LOGIT_P' : utils.logit(p)
                           })

        print "Problem summary:"
        print "> sigma^2_const    : %.8f" % sigma2_const_LM
        print "> sigma^2_pca      : %.8f" % sigma2_pca_LM
        print "> sigma^2_full     : %.8f%s" % (sigma2_full_LM, ['', ' (obs <= cov)'][n <= self.d])
        print "> number of obs    : %d" % n
        print "> number of covs   : %d (%d constraints)" % (self.d, len(param['INTERACTIONS']))
        print "> number of pcs    : %d" % self.param['DATA_PCA']
        print "> logit(p) penalty : %.8f" % self.param['LOGIT_P']


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
        if self.type == 're':
            return len(self.param['GROUPS'])
        else:
            return self.param['XtX'].shape[0] - self.param['DATA_PCA']

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

def _lpmf_re(gamma, param):
    """ 
        Log-posterior probability mass function in a Random Effect model.
        @param gamma binary vector
        @param param parameters
        @return log-probabilities
    """

    # fixed variance
    sigma2 = 1.0

    # unpack parameters
    n, d, groups, Y = [param[k] for k in ['n', 'd', 'GROUPS', 'Y']]

    # number of models
    size = gamma.shape[0]
    # array to store results
    L = numpy.empty(size, dtype=float)

    for k in xrange(size):
        # add up kinship matrices
        K = sigma2 * numpy.eye(n)
        for i in xrange(d):
            if gamma[k][i]:
                K += groups[i]

        # compute normal likelihood
        C = scipy.linalg.cholesky(K)
        b = scipy.linalg.solve(C.T, Y)
        log_diag_C = 2 * numpy.log(C.diagonal()).sum()
        L[k] = -0.5 * (log_diag_C + numpy.dot(b, b.T))
    return L


def _lpmf_hb(gamma, param):
    """ 
        Log-posterior probability mass function in a Hierarchical Bayes model.
        @param gamma binary vector
        @param param parameters
        @return log-probabilities
    """

    # unpack parameters
    penalty, XtY, XtX, c1, c2, c3, logit_p, one_over_v2, pca, interactions = \
        [param[key] for key in ['penalty', 'XtY', 'XtX', 'c1', 'c2', 'c3', 'LOGIT_P', 'one_over_v2', 'DATA_PCA', 'INTERACTIONS']]

    del param

    # number of models
    size = gamma.shape[0]
    # array to store results
    L = numpy.empty(size, dtype=float)
    gamma_pca = numpy.ones(shape=gamma.shape[1] + pca, dtype=bool)

    #total_mec_violations = 0
    for k in xrange(size):
        # model dimension
        gamma_pca[pca:] = gamma[k]
        d = gamma_pca.sum()
        
        # check main effects constraints
        if interactions.shape[0] > 0:
            mec_violations = (gamma_pca[interactions[:, 0]] > gamma_pca[interactions[:, 1]] * gamma_pca[interactions[:, 2]]).sum()
        else:
            mec_violations = False
        
        if mec_violations > 0 or d == 0:
            # unfeasible model
            L[k] = mec_violations * penalty
            #total_mec_violations += 1
        else:
            # regular model
            C = scipy.linalg.cholesky(XtX[gamma_pca, :][:, gamma_pca] + one_over_v2 * numpy.eye(d))
            if C.shape == (1, 1): b = XtY[gamma_pca, :] / float(C)
            else:                 b = scipy.linalg.solve(C.T, XtY[gamma_pca, :])
            log_diag_C = numpy.log(C.diagonal()).sum()
            L[k] = -log_diag_C - c1 * d - c2 * numpy.log(c3 - numpy.dot(b, b.T)) + d * logit_p
    #print '\nMEC VIOLATIONS', total_mec_violations / float(size)
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
