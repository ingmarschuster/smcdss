#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Posterior distribution of a model selection problem.
"""

"""
@namespace binary.posterior_bvs
$Author: christian.a.schafer@gmail.com $
$Rev: 160 $
$Date: 2011-11-14 15:03:31 +0100 (lun., 14 nov. 2011) $
@details Reads a dataset and construct the posterior probabilities of all linear models
with variables regressed on the first column.
"""

import numpy
import scipy
import binary

class Posterior(binary.binary_model.Binary):
    """
        Variable selection criterion.
    """

    def __init__(self, y, Z, param):
        """ 
            Constructor.
            \param Y explained variable
            \param Z covariates to perform selection on
            \param parameter dictionary
        """

        binary.binary_model.Binary.__init__(self, name='posterior',
                                            longname='Posterior distribution of a Bayesian variable selection problem.')

        # normalize
        self.Z = numpy.subtract(Z, Z.mean(axis=0))
        self.y = y

        # store parameters
        self.n, d = self.Z.shape

        # maximum model size
        max_size = param['PRIOR_MODEL_MAXSIZE_HP']
        if max_size is None:
            max_size = numpy.inf
        elif isinstance(max_size, str):
            if max_size == 'n': max_size = self.n

        self.param = {'Zty':numpy.dot(self.Z.T, y),
                      'penalty':-self.n * numpy.dot(y.T, y),
                      'n':self.n,
                      'd':d,
                      'fixed':param['DATA_PCA'],
                      'interactions':param['INTERACTIONS'],
                      'constraints':len(param['INTERACTIONS']),
                      'max_size':max_size
                      }
        for key in param.keys():
            if 'PRIOR' in key: self.param.update({key:param[key]})

        {'bayes':self.__init_bayes, 'bic':self.__init_bic}[param['PRIOR_CRITERION']](param)


    def __init_bayes(self, param):
        """ 
            Setup Hierarchical Bayesian posterior.
      
            \param parameter dictionary
        """
        self.f_lpmf = _lpmf_bayes

        # prior on beta
        tau = self.param['PRIOR_VAR_DISPERSION']
        if isinstance(tau, str):
            if tau == 'n': tau = self.n

        # prior on sigma
        a = self.param['PRIOR_VAR_HP_A']
        b = self.param['PRIOR_VAR_HP_B']

        # prior on gamma
        p = self.param['PRIOR_MODEL_INCLPROB_HP']

        W = {'zellner':Posterior.__zellner,
             'independent':Posterior.__independent}[param['PRIOR_COV_MATRIX_HP']](tau, self.Z)

        # constants
        self.param.update({'W': W,
                           'a*b + TSS' : a * b + Posterior.__total_sum_of_squares(self.y),
                           '-(n-1+a)/2' :-(self.n - 1 + a) / 2.0,
                           '-log(1+tau)/2' :-numpy.log(1 + tau) / 2.0,
                           '-log(tau)/2' :-numpy.log(1 + tau) / 2.0,
                           'logit(p)' : numpy.log(p / (1.0 - p))
                           })


    def __init_bic(self, param):
        """ 
            Setup Schwarz's Criterion.
      
            \param parameter dictionary
        """
        self.f_lpmf = _lpmf_bic

        # constants
        self.param.update({'W': numpy.dot(self.Z.T, self.Z) + 1e-10 * numpy.eye(self.Z.shape[1]),
                           'TSS' : Posterior.__total_sum_of_squares(self.y),
                           '-n/2' :-self.n / 2.0,
                           'log(n)' :-numpy.log(self.n) / 2.0,
                           'logit(p)':0
                           })


    def __str__(self):

        d, n, Zty, fixed = [self.param[key] for key in ['d', 'n', 'Zty', 'fixed']]
        ZtZ = numpy.dot(self.Z.T, self.Z) + 1e-10 * numpy.eye(d)
        yty = numpy.dot(self.y.T, self.y)

        # null model
        sigma2_null = Posterior.__total_sum_of_squares(self.y) / float(n - 2)

        # only fixed components
        if fixed > 0:
            gamma_pc = numpy.zeros(d, dtype=bool)
            gamma_pc[:fixed] = True
            sigma2_fixed = (yty - numpy.dot(Zty[gamma_pc, :],
                                            scipy.linalg.solve(ZtZ[gamma_pc, :][:, gamma_pc], Zty[gamma_pc, :], sym_pos=True))
                            ) / float(n - 2)
        else:
            sigma2_fixed = sigma2_null

        # full model
        if n > d:
            sigma2_full = (Posterior.__total_sum_of_squares(self.y)
                           - numpy.dot(Zty, scipy.linalg.solve(ZtZ, Zty, sym_pos=True))) / float(n - 2)
        else:
            sigma2_full = 0.0

        template = """Problem summary:
                    > sigma^2_null     : %(sigma2_null)f
                    > sigma^2_fixed    : %(sigma2_fixed)f
                    > sigma^2_full     : %(sigma2_full)f
                    > number of obs    : %(n)d
                    > number of covs   : %(d)d (%(constraints)d constraints)
                    > number of pcs    : %(fixed)d
                    > logit(p) penalty : %(logit(p))f""".replace(20 * ' ', '')

        args = dict(sigma2_null=sigma2_null, sigma2_full=sigma2_full, sigma2_fixed=sigma2_fixed)
        args.update(self.param)
        return template % args

    def getD(self):
        """ Get dimension of the variable selection problem.
            \return dimension 
        """
        return self.param['d'] - self.param['fixed']

    @staticmethod
    def __total_sum_of_squares(y):
        return numpy.dot(y.T, y) - (y.sum() ** 2) / float(y.shape[0])

    @staticmethod
    def __zellner(tau, Z):
        return (1.0 + 1.0 / tau) * numpy.dot(Z.T, Z) + 1e-10 * numpy.eye(Z.shape[1])

    @staticmethod
    def __independent(tau, Z):
        return numpy.dot(Z.T, Z) + (1.0 / tau) * numpy.eye(Z.shape[1])


#------------------------------------------------------------------------------ 


def _lpmf_bayes(gamma, param):
    """ 
        Log-posterior probability mass function in a hierarchical Bayesian model.
        
        \param gamma binary vector
        \param param parameters
        \return log-probabilities
    """

    # unpack some parameters
    penalty, W, Zty, max_size, fixed, constraints, interactions = \
        [param[key] for key in ['penalty', 'W', 'Zty', 'max_size', 'fixed', 'constraints', 'interactions']]

    # number of models
    size = gamma.shape[0]

    # array to store results
    L = numpy.empty(size, dtype=float)
    gamma_pc = numpy.ones(shape=gamma.shape[1] + fixed, dtype=bool)

    for k in xrange(size):

        # model dimension
        gamma_pc[fixed:] = gamma[k]
        gamma_size = gamma_pc.sum()

        # check main effects constraints
        if constraints > 0:
            mec_violations = (gamma_pc[interactions[:, 0]] > gamma_pc[interactions[:, 1]] * gamma_pc[interactions[:, 2]]).sum()
        else:
            mec_violations = False

        if mec_violations > 0 or gamma_size > max_size:
            # inadmissible model
            L[k] = penalty - mec_violations - gamma_size
        else:
            # regular model
            if gamma_size == 0:
                btb = 0.0
            else:
                chol = scipy.linalg.cholesky(W[gamma_pc, :][:, gamma_pc])
                b = scipy.linalg.solve(chol.T, Zty[gamma_pc, :])
                btb = numpy.dot(b.T, b)

            L[k] = param['-(n-1+a)/2'] * numpy.log(param['a*b + TSS'] - btb) + gamma_size * param['logit(p)']

            # difference between Zellner's and independent prior
            if param['PRIOR_COV_MATRIX_HP'] == 'zellner':
                L[k] += gamma_size * param['-log(1+tau)/2']
            else:
                L[k] += gamma_size * param['-log(tau)/2']
                if gamma_size > 0: L[k] -= numpy.log(chol.diagonal()).sum()

    return L


def _lpmf_bic(gamma, param):
    """ 
        Score of Schwarz's Criterion.
        
        \param gamma binary vector
        \param param parameters
        \return bic
    """

    # unpack parameters
    penalty, W, Zty, max_size, fixed, constraints, interactions = \
            [param[key] for key in ['penalty', 'W', 'Zty', 'max_size', 'fixed', 'constraints', 'interactions']]

    # number of models
    size = gamma.shape[0]

    # array to store results
    L = numpy.empty(size, dtype=float)
    gamma_pc = numpy.ones(shape=gamma.shape[1] + fixed, dtype=bool)

    for k in xrange(size):

        # model dimension
        gamma_pc[fixed:] = gamma[k]
        gamma_size = gamma_pc.sum()

        # check main effects constraints
        if constraints > 0:
            mec_violations = (gamma_pc[interactions[:, 0]] > gamma_pc[interactions[:, 1]] * gamma_pc[interactions[:, 2]]).sum()
        else:
            mec_violations = False

        if mec_violations > 0 or gamma_size > max_size:
            # inadmissible model
            L[k] = penalty - mec_violations - max_size
        else:
            # regular model
            if gamma_size > 0:
                btb = numpy.dot(Zty[gamma_pc], scipy.linalg.solve(W[gamma_pc, :][:, gamma_pc], Zty[gamma_pc, :], sym_pos=True))
            else:
                btb = 0

            L[k] = param['-n/2'] * numpy.log(param['TSS'] - btb) + gamma_size * param['log(n)']

    return L
