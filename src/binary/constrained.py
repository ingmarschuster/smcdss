#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family uniform on certain subsets. """

"""
@namespace binary.uniform
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

import numpy

import binary.base
import binary.wrapper
import scipy.special

def log_binomial(a, b):
    return (scipy.special.gammaln(a + 1) -
            scipy.special.gammaln(b + 1) -
            scipy.special.gammaln(a - b + 1))

class ConstrBinary(binary.base.BaseBinary):
    """ Binary parametric family constrained to parametric subsets. """

    def __init__(self, d, p=0.5, py_wrapper=None, name='constrained family', long_name=__doc__):
        """ 
            Constructor.
            \param d dimension
            \param p marginal probability
            \param name name
            \param long_name long_name
        """

        binary.base.BaseBinary.__init__(self, py_wrapper=py_wrapper, name=name, long_name=long_name)

        self.param.update({'d':d, 'logit_p':numpy.log(p / (1 - p)), 'p':p})
        self.pp_modules = ('numpy', 'binary.constrained')


    def _getD(self):
        """ Get dimension of instance. \return dimension """
        return self.param['d']


class ConstrSizeBinary(ConstrBinary):
    """ Binary parametric family constrained to vectors of norm not greater than q subsets. """

    def __init__(self, d, q, p=0.5, py_wrapper=None, name='constrained size family', long_name=__doc__):
        """ 
            Constructor.
            \param d dimension
            \param q maximum size
            \param p marginal probability
            \param name name
            \param long_name long_name
        """

        if py_wrapper is None: py_wrapper = binary.wrapper.constrained_size()

        ConstrBinary.__init__(self, d=d, p=p, py_wrapper=py_wrapper, name=name, long_name=long_name)

        # compute binomial probabilities up to q
        m = numpy.empty(q + 1, dtype=float)
        for k in xrange(q + 1):
            m[k] = log_binomial(d, k) + k * self.param['logit_p']

        # deal with sum of exponentials
        m = numpy.exp(m - m.max())
        m /= m.sum()

        self.param.update({'q':q, 'm_cumsum':m.cumsum()})

    def __str__(self):
        return 'maximum size: %d' % self.param['q']

    @classmethod
    def _lpmf(cls, gamma, param):
        """ 
            Log-probability mass function.
            \param gamma binary vector
            \param param parameters
            \return log-probabilities
        """
        L = -numpy.inf * numpy.ones(gamma.shape[0])
        index = numpy.array(gamma.sum(axis=1), dtype=float)
        index = index[index <= param['q']]
        L[gamma.sum(axis=1) <= param['q']] = index * param['logit_p']
        return L

    @classmethod
    def _rvs(cls, U, param):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        Y = numpy.zeros((U.shape[0], U.shape[1]), dtype=bool)
        d, m_cumsum = param['d'], param['m_cumsum']

        for k in xrange(U.shape[0]):
            perm = numpy.arange(d)
            for i in xrange(d):
                # pick an element in p[:i+1] with which to exchange p[i]
                j = int(U[k][i] * (d - i))
                perm[d - 1 - i], perm[j] = perm[j], perm[d - 1 - i]
            # draw the number of nonzero elements

            for r, p in enumerate(m_cumsum):
                if U[k][d - 1] < p: break

            Y[k][perm[:r]] = True
        return Y

    @classmethod
    def random(cls, d, p=None):
        """ 
            Construct a random family for testing.
            \param cls class
            \param d dimension
        """
        # random marginal probability
        if p is None: p = 0.01 + 0.98 * numpy.random.random()
        # random maximal norm
        q = numpy.random.randint(d) + 1
        return cls(d=d, q=q, p=p)

    def _getMean(self):
        mean = self.param['p'] * self.param['q'] / float(self.param['d'])
        return mean * numpy.ones(self.param['d'])


class ConstrInteractionBinary(ConstrBinary):
    """ Binary parametric family constrained to vectors that verify certain interactions. """

    def __init__(self, d, constrained=None, p=0.5, py_wrapper=None, name='uniform family', long_name=__doc__):
        """ 
            Constructor.
            \param d dimension
            \param constraints list
            \param name name
            \param long_name long_name
        """

        if py_wrapper is None: py_wrapper = binary.wrapper.constrained_interaction()
        ConstrBinary.__init__(self, d=d, p=p, py_wrapper=py_wrapper, name=name, long_name=long_name)

        free = [i for i in xrange(d) if not i in [item for sublist in constrained for item in sublist]]
        self.param.update({'constrained':constrained, 'free':free})

    def __str__(self):
        return 'constraints:\n' + str(self.param['constrained']) + '\n'

    @classmethod
    def _lpmf(cls, Y, param):
        """ 
            Log-probability mass function.
            \param gamma binary vector
            \param param parameters
            \return log-probabilities
        """
        L = Y.sum(axis=1) * param['logit_p']
        violations = ConstrInteractionBinary.constraint_violation(Y, param)
        L[violations] = -numpy.inf
        return L

    @classmethod
    def constraint_violation(cls, Y, param):
        constrained = param['constrained']
        V = numpy.zeros(Y.shape[0], dtype=bool)

        for k in xrange(Y.shape[0]):
            for bundle in constrained:
                if Y[k, bundle[0]] and not Y[k, bundle[1:]].prod():
                    V[k] = True
                    break
        return V

    @classmethod
    def _rvs(cls, U, param):
        """ 
            Generates a random variable.
            \param U uniform variables
            \param param parameters
            \return binary variables
        """
        Y = numpy.zeros((U.shape[0], U.shape[1]), dtype=bool)

        p, constrained, free = param['p'], param['constrained'], param['free']

        for k in xrange(U.shape[0]):
            Y[k][free] = U[k][free] < p
            for bundle in constrained:
                index = list(bundle)
                n = len(bundle)
                if U[k, bundle[0]] < p ** n / (1.0 - p + p ** n):
                    Y[k][index] = numpy.ones(n, dtype=bool)
                else:
                    Y[k][index] = U[k][index] < p
                    Y[k, bundle[0]] = False
        return Y

    @classmethod
    def random(cls, d, p=None):
        """ 
            Construct a random family for testing.
            \param cls class
            \param d dimension
        """
        # random marginal probability
        if p is None: p = 0.01 + 0.98 * numpy.random.random()
        perm = numpy.arange(d, dtype=int)
        # random permutation
        for i in xrange(d):
            j = int(numpy.random.random() * (d - i))
            perm[d - 1 - i], perm[j] = perm[j], perm[d - 1 - i]
        constrained = []
        # random constraints
        i = 0
        while True:
            n = numpy.random.randint(low=2, high=5)
            if i + n > d:break
            constrained += [(numpy.array(perm[i:(i + n)]))]
            i += n
        return cls(d=d, constrained=constrained, p=p)


    def _getMean(self):
        mean = 0.5 * self.param['q'] / float(self.param['d'])
        return mean * numpy.ones(self.param['d'])

    def _getD(self):
        """ Get dimension of instance. \return dimension """
        return self.param['d']
