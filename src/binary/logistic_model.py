'''
@author cschafer
'''

from time import clock
from auxpy.data import *
from binary import productBinary
from numpy import *
from scipy.weave import inline, converters
from platform import system
from scipy.linalg import solve

CONST_PRECISION = 0.00001
CONST_ITERATIONS = 30


if system() == 'Linux':    hasWeave = True
else:                      hasWeave = False


class logisticRegrBinary(productBinary):
    '''
        A multivariate Bernoulli with conditionals based on logistic regression models.
    '''

    def __init__(self, Beta):
        '''
            Constructor.
            @param Beta matrix of regression coefficients
        '''

        ## matrix of regression coefficients 
        self.Beta = Beta

        productBinary.__init__(self, name='logistic-regression-binary', longname='A multivariate Bernoulli with conditionals based on logistic regression models.')

    @classmethod
    def independent(cls, p):
        '''
            Constructs a hidden-normal-binary model with independent components.
            @param cls class 
            @param p mean
        '''
        d = p.shape[0]
        p[1:] = log(p[1:] / (1 - p[1:]))
        Beta = zeros((d, d))
        Beta[:, 0] = p
        return cls(Beta)

    @classmethod
    def random(cls, d):
        '''
            Constructs a random logistic-regression-binary model for testing.
            @param cls class 
            @param d dimension
        '''
        cls = logisticRegrBinary.independent(random.random(d))
        cls.Beta[:, 1:] = random.normal(scale=3.0, size=(d, d - 1))
        return cls

    @classmethod
    def fromData(cls, sample):
        '''
            Construct a product-binary model from data.
            @param cls class
            @param sample a sample of binary data
        '''
        return cls(calcBeta(sample))

    def _pmf(self, gamma):
        '''
            Probability mass function.
            @param gamma: binary vector
        '''
        return exp(self._lpmf(gamma))

    def _lpmf(self, gamma):
        '''
            Log-probability mass function.
            @param gamma binary vector    
        '''
        if hasWeave:
            return self.__lpmf_weave(gamma)
        else:
            return self.__lpmf_python(gamma)

    def _rvs(self):
        '''
            Samples from the model.
            @return random variate
        '''
        if hasWeave:
            return self.__rvs_weave()[0]
        else:
            return self.__rvs_python()[0]

    def _rvslpmf(self):
        '''
            Samples from the model and evaluates the likelihood of the sample.
            @return random variate
            @return likelihood
        '''
        if hasWeave:
            return self.__rvs_weave()
        else:
            return self.__rvs_python()

    def getD(self):
        '''
            Get dimension.
            @return dimension 
        '''
        return self.Beta.shape[0]

    def __str__(self):
        return format_matrix(self.Beta, 'Beta')

    def __lpmf_weave(self, gamma):
        Beta = self.Beta
        d = Beta.shape[0]
        logvprob = empty(1, dtype=float)
        code = \
        """
        double sum, logcprob;
        int i,j;
               
        if (gamma(0)) logvprob = log(Beta(0,0));
        else logvprob = log(1-Beta(0,0));
        
        for(i=1; i<d; i++){
        
            /* Compute log conditional probability that gamma(i) is one */
            sum = Beta(i,0);
            for(j=1; j<=i; j++){        
                sum += Beta(i,j) * gamma(j-1);
            }
            logcprob = -log(1+exp(-sum));
            
            /* Compute log conditional probability of whole gamma vector */
            logvprob += logcprob;        
            if (!gamma(i)) logvprob -= sum;
            
        }
        """
        inline(code, ['d', 'Beta', 'gamma', 'logvprob'], \
                     type_converters=converters.blitz, compiler='gcc')
        return float(logvprob)

    def __lpmf_python(self, gamma):
        Beta = self.Beta
        d = Beta.shape[0]

        if gamma[0]: logvprob = log(Beta[0][0])
        else: logvprob = log(1 - Beta[0][0])

        # Compute log conditional probability that gamma(i) is one for i > 0
        sum = Beta[1:, 0].copy()
        for i in range(1, d): sum[i - 1] += dot(Beta[i, 1:i + 1], gamma[0:i])
        logcprob = -log(1 + exp(-sum))

        # Compute log conditional probability of whole gamma vector
        logvprob += logcprob.sum() - sum[-gamma[1:]].sum()
        return logvprob

    def __rvs_weave(self):
        Beta = self.Beta
        d = Beta.shape[0]
        u = random.rand(d)
        gamma = empty(d, dtype=bool)
        logvprob = empty(1, dtype=float)
        code = \
        """
        double sum, logcprob;
        int i,j;
        
        /* Draw an independent gamma(0) */
        gamma(0) = (u(0) < Beta(0,0));
        
        if (gamma(0)) logvprob = log(Beta(0,0));
        else logvprob = log(1-Beta(0,0));
        
        for(i=1; i<d; i++){
        
            /* Compute log conditional probability that gamma(i) is one */
            sum = Beta(i,0);
            for(j=1; j<=i; j++){        
                sum += Beta(i,j) * gamma(j-1);
            }
            logcprob = -log(1+exp(-sum));
            
            /* Generate the ith entry */
            gamma(i) = (log(u(i)) < logcprob);
            
            /* Compute log conditional probability of whole gamma vector */
            logvprob += logcprob;        
            if (!gamma(i)) logvprob -= sum;
            
        }
        """
        inline(code, ['d', 'u', 'Beta', 'gamma', 'logvprob'], \
                     type_converters=converters.blitz, compiler='gcc')
        return gamma, logvprob

    def __rvs_python(self):
        Beta = self.Beta
        d = Beta.shape[0]
        gamma = empty(d, dtype=bool)
        logu = log(random.rand(d))

        # Draw an independent gamma[0]
        gamma[0] = random.rand() < Beta[0][0]
        if gamma[0]: logvprob = log(Beta[0][0])
        else: logvprob = log(1 - Beta[0][0])

        for i in range(1, d):
            # Compute log conditional probability that gamma(i) is one
            sum = Beta[i][0] + dot(Beta[i, 1:i + 1], gamma[0:i])
            logcprob = -log(1 + exp(-sum))

            # Generate the ith entry
            gamma[i] = logu[i] < logcprob

            # Compute log conditional probability of whole gamma vector
            logvprob += logcprob
            if not gamma[i]: logvprob -= sum

        return gamma, logvprob

    d = property(fget=getD, doc="dimension")

def calcBeta(sample, Init=None, verbose=False):
    '''
        Computes the logistic regression coefficients of all conditionals. 
        @param sample binary data
        @param Init matrix with inital values
        @param verbose print to stdout 
        @return matrix of regression coefficients
    '''

    if sample.d == 0: return array([])
    
    t = clock()
    n = sample.size
    d = sample.d

    # Add constant column.
    X = column_stack((ones(n, dtype=bool)[:, newaxis], sample.procData(dtype=bool)))
    if sample.isWeighted: XW = sample.w[:, newaxis] * X
    else: XW = X

    if Init == None: Init = zeros((d, d))
    Beta = zeros((d, d), dtype=float)
    Beta[0][0] = sum(X[:, 1]) / float(n)

    # Loop over all dimensions compute logistic regressions.
    resp = array([0, 0])
    for m in range(1, d):

        Beta[m, :m + 1], r = calcLogRegr(y=X[:, m + 1], X=X[:, :m + 1], XW=XW[:, :m + 1], init=Init[m, :m + 1])
        resp += r

    if verbose: print 'loops %.3f, failures %i, time %.3f' % (resp[0] / float(d - 1), resp[1], clock() - t)

    return Beta

def calcLogRegr(y, X, XW=None, init=None):
    '''
        Computes the logistic regression coefficients.. 
        @param y explained variable
        @param X covariates
        @param X weighted covariates
        @param init initial value
        @return vector of regression coefficients
    '''
    n = X.shape[0]
    d = X.shape[1]

    if init == None: beta = zeros(d)
    else:            beta = init

    for iter in range(CONST_ITERATIONS):

        last_beta = beta.copy()

        if hasWeave:
            v = empty(n)
            P = empty(n);
            code = \
            """
            double p, Xbeta;
            
            for (int i = 0; i < n; i++)
            {
                Xbeta = 0;
                for(int k = 0; k <= d; k++)
                {
                    Xbeta += X(i,k) * beta(k);
                }
                p = 1 / (1 + exp(-Xbeta));
                P(i) = p * (1-p);
                v(i) = P(i) * Xbeta + y(i) - p;
            }
            """
            inline(code, ['beta', 'X', 'y', 'P', 'd', 'n', 'v'], \
            type_converters=converters.blitz, compiler='gcc')
        else:
            Xbeta = dot(X, beta)
            p = pow(1 + exp(-Xbeta), -1)
            P = p * (1 - p)
            v = P * Xbeta + y - p

        XWDX = dot(XW.T, P[:, newaxis] * X) + exp(-10) * eye(d)

        # Solve Newton-Raphson equation.
        beta = solve(XWDX, dot(XW.T, v), sym_pos=True)

        if (abs(last_beta - beta) < CONST_PRECISION).all(): break

    # Mark as failed, if the Newton-Raphson iteration did not converge.
    failed = (iter == CONST_ITERATIONS)
    if failed:
        beta = zeros(d)
        p = y.sum() / float(n)
        beta[0] = log(p / (1 - p))

    return beta, (iter, failed)
