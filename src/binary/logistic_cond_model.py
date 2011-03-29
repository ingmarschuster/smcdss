#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @Author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from binary import *

class LogisticBinary(product_model.ProductBinary):
    ''' A binary model with conditionals based on logistic regressions. '''

    def __init__(self, Beta, name='logistic binary',
                             longname='A binary model with logistic conditionals.'):
        ''' Constructor.
            @param Beta matrix of regression coefficients
        '''

        product_model.ProductBinary.__init__(self, p=utils.inv_logit(numpy.diagonal(Beta)),\
                                             name=name, longname=longname)

        if 'cython' in utils.opts:
            self.f_rvslpmf = utils.cython.logistic_rvslpmf
            self.f_lpmf = utils.cython.logistic_lpmf
            self.f_rvs = utils.cython.logistic_rvs
        else:
            self.f_rvslpmf = utils.python.logistic_rvslpmf
            self.f_lpmf = utils.python.logistic_lpmf
            self.f_rvs = utils.python.logistic_rvs

        self.param = dict(Beta=Beta)

    @classmethod
    def independent(cls, p):
        ''' Constructs a logistic binary model with independent components.
            @param cls instance
            @param p mean
            @return logistic model
        '''
        d = p.shape[0]
        Beta = numpy.diag(numpy.log(p / (1 - p)))
        return cls(Beta)

    @classmethod
    def uniform(cls, d):
        ''' Constructs a uniform logistic binary model.
            @param cls instance
            @param d dimension
            @return logistic model
        '''
        Beta = numpy.zeros((d, d))
        return cls(Beta)

    @classmethod
    def random(cls, d, dep=3.0):
        ''' Constructs a random logistic binary model.
            @param cls instance
            @param d dimension
            @param dep strength of dependencies [0,inf)
            @return logistic model
        '''
        cls = LogisticBinary.independent(p=numpy.random.random(d))
        Beta = numpy.random.normal(scale=dep, size=(d, d))
        Beta *= numpy.dot(Beta, Beta)
        for i in xrange(d): Beta[i, i] = cls.param['Beta'][i, i]
        cls.param['Beta'] = Beta
        return cls

    @classmethod
    def from_data(cls, sample, Init=None, job_server=None, eps=0.02, delta=0.075, verbose=False):
        ''' Construct a logistic-regression binary model from data.
            @param cls instance
            @param sample a sample of binary data
            @param Init matrix with inital values
            @param eps marginal probs in [eps,1-eps] > logistic model
            @param delta abs correlation in  [delta,1] > association
            @param xi marginal probs in [xi,1-xi] > random component
            @param verbose detailed output
            @return logistic model
        '''
        return cls(calc_Beta(sample, Init=Init, job_server=job_server, eps=eps, delta=delta, verbose=verbose))

    def renew_from_data(self, sample, job_server=None, eps=0.02, delta=0.075, lag=0, verbose=False):
        ''' Construct a logistic-regression binary model from data.
            @param sample a sample of binary data
            @param eps marginal probs in [eps,1-eps] > logistic model
            @param delta abs correlation in  [delta,1] > association
            @param xi marginal probs in [xi,1-xi] > random component
            @param verbose detailed output
            @return logistic model
        '''
        # Compute new parameter from data.
        newBeta = calc_Beta(sample=sample, Init=self.param['Beta'], job_server=job_server, eps=eps, delta=delta, verbose=verbose)

        # Set convex combination of old and new parameter.
        self.param['Beta'] = (1 - lag) * newBeta + lag * self.param['Beta']

    @classmethod
    def from_loglinear_model(cls, llmodel):
        ''' Constructs a logistic model that approximates a log-linear model.
            @param cls instance
            @param llmodel log-linear model
            @todo Instead of margining out in arbitrary order, we could use a greedy approach
            which picks the next dimension by minimizing the error made in the Taylor approximation. 
        '''
        d = llmodel.d
        Beta = numpy.zeros((d, d))
        Beta[0, 0] = utils.logit(llmodel.p_0)

        A = numpy.copy(llmodel.A)
        for i in xrange(d - 1, 0, -1):
            Beta[i, 0:i] = A[i, :i] * 2.0
            Beta[i, i] = A[i, i]
            A = qu_exponential_model.calc_marginal(A)

        return cls(Beta)

    def getD(self):
        ''' Get dimension.
            @return dimension 
        '''
        return self.param['Beta'].shape[0]

    def getModelSize(self):
        ''' Get ratio of used parameters over d*(d+1)/2.
            @return dimension 
        '''
        return '%i/%i' % ((self.Beta <> 0.0).sum(), self.d * (self.d + 1) / 2.0)

    def __str__(self):
        return utils.format.format_matrix(self.param['Beta'], 'Beta')

    d = property(fget=getD, doc="dimension")

def calc_Beta(sample, eps=0.02, delta=0.05, Init=None, job_server=None, negative_weights=False, verbose=True):
    ''' Computes the logistic regression coefficients of all conditionals. 
        @param sample binary data
        @param eps marginal probs in [eps,1-eps] > logistic model
        @param delta abs correlation in  [delta,1] > association
        @param Init matrix with initial values
        @param job_server job server
        @param verbose print to stdout 
        @return matrix of regression coefficients
    '''

    if sample.d == 0: return numpy.array([])

    t = time.time()
    n = sample.size
    d = sample.d

    X = numpy.column_stack((sample.proc_data(dtype=float), numpy.ones(n, dtype=float)[:, numpy.newaxis]))

    # Compute weighted sample.
    if negative_weights:
        w = numpy.array(sample._W); w /= w.sum()
        XW = w[:, numpy.newaxis] * X
    else:
        w = numpy.array(sample.nW)
        XW = w[:, numpy.newaxis] * X

    # Compute slightly adjusted mean and real log odds.
    p = 1e-10 * 0.5 + (1.0 - 1e-10) * XW[:, 0:d].sum(axis=0)

    # Assure that p is in the unit interval.
    if negative_weights: p = numpy.array([max(min(x, 1.0 - 1e-10), 1e-10) for x in p])

    # Compute logits.
    logit = utils.logit(p)

    # Find strong associations.
    L = abs(utils.data.calc_cor(X[:, 0:d], w=w)) > delta

    # Remove components with extreme marginals.
    for i, m in enumerate(p):
        if m < eps or m > 1.0 - eps:
            L[i, :] = numpy.zeros(d, dtype=bool)

    # Initialize Beta with logits on diagonal.
    Beta = numpy.zeros((d, d))
    if Init is None: Init = numpy.diag(logit)
    
    if verbose: stats = dict(regressions=0.0, failures=0, iterations=0, product=d, logistic=0)

    # Loop over all dimensions compute logistic regressions.
    jobs = list()
    for i in range(d):
        l = list(numpy.where(L[i, 0:i])[0])
        if len(l) > 0:
            if not job_server is None:
                jobs.append([i, l + [i],
                        (job_server.submit(
                        func=calc_log_regr,
                        args=(X[:, i], X[:, l + [d]], XW[:, l + [d]], Init[i, l + [i]], w, False),
                        modules=('numpy', 'scipy.linalg', 'binary')))
                ])
            else:
                jobs.append([i, l + [i], (calc_log_regr(
                              X[:, i], X[:, l + [d]], XW[:, l + [d]], Init[i, l + [i]], w, False))
                ])

        else:
            Beta[i, i] = logit[i]

    # wait and retrieve results
    if not job_server is None: job_server.wait()

    # write results to Beta matrix
    for i, l, job in jobs:
        if isinstance(job, tuple): beta, iterations = job
        else: beta, iterations = job()

        if verbose:
            stats['iterations'] = iterations
            stats['regressions'] += 1.0
            stats['product'] -= 1
            stats['logistic'] += 1

        if not beta is None:
            Beta[i, l] = beta
        else:
            if verbose: stats['failures'] += 1
            Beta[i, i] = logit[i]

    if verbose:
        stats.update({'time':time.time() - t})
        if stats['regressions'] > 0: stats['iterations'] /= stats['regressions']
        print 'Logistic model: (p %(product)i, l %(logistic)i), loops %(iterations).3f, failures %(failures)i, time %(time).3f\n' % stats

    return Beta


def calc_log_regr(y, X, XW, init, w=None, verbose=False):
    '''
        Computes the logistic regression coefficients.. 
        @param y explained variable
        @param X covariables
        @param X weighted covariables
        @param init initial value
        @return vector of regression coefficients
    '''

    # Initialize variables. 
    n = X.shape[0]
    d = X.shape[1]
    beta = init
    if w is None: w = numpy.ones(n) / float(n)
    v = numpy.empty(n)
    P = numpy.empty(n)
    llh = -numpy.inf
    _lambda = 1e-8

    for i in range(CONST_ITERATIONS):

        # Save last iterations values.
        last_llh = llh
        last_beta = beta.copy()

        # Compute Fisher information at beta
        Xbeta = numpy.dot(X, beta)
        p = numpy.power(1 + numpy.exp(-Xbeta), -1)
        P = p * (1 - p)
        XWPX = numpy.dot(XW.T, P[:, numpy.newaxis] * X) + _lambda * numpy.eye(d)
        v = P * Xbeta + y - p

        # Solve Newton-Raphson equation.
        try:
            beta = scipy.linalg.solve(XWPX, numpy.dot(XW.T, v), sym_pos=True)
        except:
            if verbose: print '> likelihood not unimodal'
            beta = scipy.linalg.solve(XWPX, numpy.dot(XW.T, v), sym_pos=False)

        # Compute the log-likelihood.
        llh = -0.5 * _lambda * numpy.dot(beta, beta) + (w * (y * Xbeta + numpy.log(1 + numpy.exp(Xbeta)))).sum()

        if abs(beta).max() > 1e4:
            if verbose: print 'convergence failure\n'
            return None, i

        if (abs(last_beta - beta) < CONST_PRECISION).all():
            if verbose: print 'no change in beta\n'
            break

        if verbose: print '%i) log-likelihood %.2f' % (i + 1, llh)
        if abs(last_llh - llh) < CONST_PRECISION:
            if verbose: print 'no change in likelihood\n'
            break

    return beta, i + 1

def main():
    x=LogisticBinary.random(5)
    print x.Beta
    print x.r

if __name__ == "__main__":
    main()


#        if len(adjIndex) == 0:
#            self.Beta = array([])
#            self.Beta.shape = (0, 0)
#            return
#
#        prvBeta = self.Beta
#        adjBeta = zeros((len(adjIndex), len(adjIndex)), dtype = float)
#
#        mapping = [(adjIndex.index(i), prvIndex.index(i)) for i in list(set(prvIndex) & set(adjIndex))]
#        for adj_x, prv_x in mapping:
#            adjBeta[adj_x, 0] = prvBeta[prv_x, 0]
#            for adj_y, prv_y in mapping:
#                if adj_y >= len(adjIndex) - 1 or prv_y >= len(prvIndex) - 1: continue
#                adjBeta[adj_x, adj_y + 1] = prvBeta[prv_x, prv_y + 1]
