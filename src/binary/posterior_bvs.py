'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from binary import *
from csv import reader

class posteriorBinary(productBinary):
    '''
        Reads a dataset and construct the posterior probabilities of all linear models
        with variates regressed on the first column.
    '''
    def __init__(self, dataFile, modelType='hb'):
        '''
            Constructor.
            @param dataFile data file
            @param modelType Hierachical Bayesian (hb) or Bayesian Information Criterion (bic)
        '''

        productBinary.__init__(self, name='posterior-binary', longname='A posterior distribution of a Bayesian variable selection problem.')

        # import dataset
        dataReader = reader(open(dataFile), delimiter=';')
        X = []; Y = []
        for row in dataReader:
            # TODO: Open for all columns
            X.append(array([float(entry) for entry in row[74:78]]))
            Y.append(float(row[0]))
        X = array(X); Y = array(Y)
        ## data file
        self.dataFile = dataFile

        ## Hierachical Bayesian (hb) or Bayesian Information Criterion (bic)
        self.modelType = modelType
        ## sample size
        self.n = X.shape[0]
        ## X^t times y
        self.XtY = dot(X.T, Y)
        ## X^t times X
        self.XtX = dot(X.T, X)

        #-------------------------------------------------- Hierachical Bayesian
        if self.modelType == 'hb':
            # variance of betas
            v1 = 100
            # inverse gamma prior of sigma^2
            lambda_ = 1
            # inverse gamma prior of sigma^2
            nu_ = .1
            ## constant 1
            self.c1 = 0.5 * log(v1)
            ## constant 2
            self.c2 = 0.5 * (self.n + nu_)
            ## constant 3
            self.c3 = nu_ * lambda_ + dot(Y.T, Y)

        #---------------------------------------- Bayesian Information Criterion
        if self.modelType == 'bic':
            ## constant 1
            self.c1 = dot(Y.T, Y) / float(self.n)
            ## constant 2
            self.c2 = self.XtY / float(self.n)
            ## constant 3
            self.c3 = 0.5 * log(self.n)


    def hb(self, gamma):
        '''
        Evaluates the log posterior density from a conjugate hierarchical setup ([George, McCulloch 1997], simplified).
        
            gamma    a binary vector
        '''
        p = gamma.sum()
        if p == 0:
            return - inf
        else:
            K = cholesky(self.XtX[gamma, :][:, gamma] + 10 ** -8 * eye(p))
            if K.shape == (1, 1):
                w = self.XtY[gamma, :] / float(K)
            else:
                w = solve(K.T, self.XtY[gamma, :])
            k = log(K.diagonal()).sum()
        return - k - self.c1 * p - self.c2 * log(self.c3 - dot(w, w.T))

    def bic(self, gamma):
        '''
        Evaluates the Bayesian Information Criterion. 
        
            gamma    a binary vector
        '''
        p = gamma.sum()
        if p == 0:
            return - inf
        else:
            try:
                beta = solve(self.XtX[gamma, :][:, gamma], self.XtY[gamma, :], sym_pos=True)
            except:
                beta = solve(self.XtX[gamma, :][:, gamma] + 10 ** -8 * eye(p), self.XtY[gamma, :], sym_pos=True)
        return - 0.5 * self.n * log(self.c1 - dot(self.c2[gamma], beta)) - self.c3 * p

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.XtX.shape[0]

    def _lpmf(self, gamma):
        '''
            Unnormalized log-probability mass function.
            @param gamma binary vector
        '''
        if self.modelType == 'bic':
            return float(self.bic(gamma))
        else:
            return float(self.hb(gamma))

    def _pmf(self, gamma):
        '''
            Unnormalized probability mass function.
            @param gamma binary vector
        '''
        if not hasattr(self, 'loglevel'): self._explore()
        return exp(self.lpmf(gamma) - self.loglevel)

    def _rvs(self):
        '''
            Generates a sample from the posterior density rejecting from a proposals of independent 1/2-binary variables.   
        '''
        while True:
            proposal = 0.5 * ones(self.d) > rand(self.d)
            if rand() < self.pmf(proposal): return proposal

    def _explore(self):
        '''
            Find the maximmum of the log-posterior.
        '''
        ## level of logarithm
        self.loglevel = -inf
        for dec in range(2 ** self.d):
            bin = dec2bin(dec, self.d)
            eval = self.lpmf(bin)
            if eval > self.loglevel: self.loglevel = eval

    d = property(fget=getD, doc="dimension")
