'''
Created on 30 nov. 2010

@author: cschafer
'''


from numpy import *
from datetime import time
from operator import setitem
from copy import deepcopy

from binary import *
from auxpy.data import data
from amcmbs.mcmc import mcmc
from auxpy.default import dicCE, dicData

if system() == 'Linux':    hasWeave = True
else:                      hasWeave = False

class ParticleSystem(object):

    def __init__(self, d, n):
        self.d = d
        self.n = n
        self.X = resize(empty(target.p, dtype=bool), (n, d))
        self.w = zeros(n, dtype=float)
        self.log_f = empty(n, dtype=float)
        self.log_m = 2 ** -d * ones(n, dtype=float)
        self.prev_bridge = zeros(n, dtype=float)
        self.bridge = zeros(n, dtype=float)
        id = n * [0]
        rho = 0

    def getData(self):
        d = data(self.X, self.w)
        return d

def smc(self, f, verbose=True):

    start = clock()

    ps = ParticleSystem(f.d, dictSMC['n_particles'])
    model = dictSMC['model'].uniform(f.d)

    for i in range(ps.n):
        ps.X[i] = model.rvs()
        ps.log_f[i] = f.lpmf(X[i])
        ps.id[i] = getID(X[i])

    # run sequential MC scheme
    while ps.rho < 1.0:

        next_rho(ps)
        model = model.from_data(ps.getData())
        resample_system(ps)
        move_system(ps)

    # return results
    print "\nDone in %.3f seconds.\n" % (clock() - start)

def getID(x):
    return dot(k, x)

def nextRho(w, min_ess):
    ''' bisectional search to find optimal step length'''
    pass

def moveXystem(self):
    '''
    Move X according to an MH kernel to fight depletion of the particle system.
    '''
    if self.verbose: print "move..."; step = int(self.n['X'] / 10)
    previous_particleDiversity = 0
    for iter in range(10):
        acceptanceRatio = 0
        self.n['moves'] += 1
        if self.verbose:
            stdout.write("%i " % iter)
            stdout.write("[")

        # generate new X from invariant kernel
        for index in range(self.n['X']):
            if self.verbose and index % step == 0: stdout.write("-")
            acceptanceRatio += self.kernel_indmh(index)

        if self.verbose: print "]"

        particleDiversity = self.getParticleDiversity()

        # check if binary_ind performs poorly and change to binary_log
        if self.proposalDistr.name == 'binary_ind' and acceptanceRatio < self.n['X'] * 0.25:
            if self.verbose: print "switch to binary_log after next resampling..."
            self.changeProposalDistr = True

        if self.verbose:print "acc: %.3f, pdiv: %.3f" % (acceptanceRatio / float(self.n['X']), particleDiversity)
        if particleDiversity - previous_particleDiversity < 0.05 or particleDiversity > 0.92: break
        else: previous_particleDiversity = particleDiversity

    log_weights = zeros(self.n['X'])
    prev_bridge = rho * log_f + (1 - rho) * self.logprior()
    return particleDiversity


def kernel_indmh(ps, index):
    '''
    Metropolis Hasting kernel with independent proposal distribution estimated from the particle system.
    
    @param index Index of the particle to be procgetEssed by the kernel.
    '''

    # generate proposal
    proposal, new_proposalScore = self.proposalDistr.rvsplus()
    new_targetScore = self.target.lpmf(proposal)
    self.n['target evals'] += 1

    if self.kappa > 0: new_priorScore = self.logprior(self.priorDistr.lpmf(proposal))
    else: new_priorScore = self.logprior()

    new_bridgeScore = rho * new_targetScore + (1 - rho) * new_priorScore
    bridgeScore = rho * ps['logtarget'][index] + (1 - rho) * self.logprior(index)

    # compute acceptance probability and do MH step
    if rand() < exp(log_prop[index] + new_bridgeScore - new_proposalScore - bridgeScore):
        X  [index] = proposal
        self.ps['id']    [index] = bin2str(proposal)
        log_f  [index] = new_targetScore
        log_prop[index] = new_proposalScore
        if self.kappa > 0: self.ps['logprior'][index] = new_priorScore
        return 1
    else:
        return 0

def updateLogWeights(ps):
    '''
    Update the log weights. Return effective sample size.
    '''
    ps['bridge t'] = rho * ps['logtarget'] + (1 - rho) * logprior()
    log_weights += ps['bridge t'] - prev_bridge
    prev_bridge = deepcopy(ps['bridge t'])
    return self.getEss()

def normalize(ps):
    '''
    Return normalized importance weights.
    '''
    logweights = log_weights - log_weights.max();
    weights = exp(logweights); weights /= sum(weights)
    return weights

def getEss(w):
    '''
    Return effective sample size 1/(sum_{w \in weights} w^2) .
    '''
    return 1 / pow(w, 2).sum()

def getParticleDiversity(ps):
    '''
    Return the particle diversity.
    '''
    d = {}
    map(setitem, (d,)*len(ps['id']), ps['id'], [])
    return len(d.keys()) / float(dictSMC['n_X'])

def resampleXystem(ps):
    '''
    Resample the particle system.
    '''
    if self.verbose: print "resample..."
    indices = resample()

    X = X[indices]
    self.ps['id'] = [ self.ps['id'][i] for i in indices ]
    log_f = log_f[indices]
    log_prop = log_prop[indices]

    if self.verbose: print "pdiv: ", self.getParticleDiversity()

    # update log proposal/prior values - use that X are grouped after resampling
    log_prop[0] = self.proposalDistr.lpmf(X[0])
    if self.kappa > 0:
        self.ps['logprior'][0] = self.priorDistr.lpmf(X[0])
    for i in range(1, self.n['X']):
        if (log_prop[i] == log_prop[i - 1]).all():
            log_prop[i] = log_prop[i - 1]
            if self.kappa > 0:
                self.ps['logprior'][i] = self.priorDistr.lpmf(self.ps['logprior'][i - 1])
        else:
            log_prop[i] = self.proposalDistr.lpmf(X[i])
            if self.kappa > 0:
                self.ps['logprior'][i] = self.priorDistr.lpmf(X[i])

def logprior(self, x=None):
    '''
    Return the prior log probability.
    
    @param x
    If x is an index, the function returns self.ps['logprior'][x] plus the log level.
    If x is a float, it returns x plus the log level.
    If x is not specified, it returns the vector self.ps['logprior'] plus the loglevel.
    '''
    if self.kappa == 0:
        return self.loglevel - self.target.p * log(2)
    if type(x).__name__ == "int":
        return self.ps['logprior'][x] + self.loglevel
    if type(x).__name__ == "float64":
        return x + self.loglevel
    if x == None:
        return self.ps['logprior'] + self.loglevel

def resample(w):
    if hasWeave: return self.resample_weave(w)
    else: return self.resample_python(w)

def resample_python(w):
    '''
    Compute the particle indices by residual resampling - adopted from Pierre's code.
    '''
    u = random.uniform(size=1, low=0, high=1)
    cnw = self.n['X'] * cumsum(w)
    j = 0
    indices = empty(self.n['X'], dtype="int")
    for k in xrange(self.n['X']):
        while cnw[j] < u:
            j = j + 1
        indices[k] = j
        u = u + 1.
    return indices

def resample_weave(w):
    '''
    Compute the particle indices by residual resampling using scypy.weave.
    '''
    code = \
    """
    int j = 0;
    double cumsum = weights(0);
    
    for(int k = 0; k < n; k++)
    {
        while(cumsum < u)
        {
        j++;
        cumsum += weights(j);
        }
        indices(k) = j;
        u = u + 1.;
    }
    """
    u = float(random.uniform(size=1, low=0, high=1)[0])
    n = self.n['X']
    weights = n * w

    indices = zeros(self.n['X'], dtype="int")
    weave.inline(code, ['u', 'n', 'weights', 'indices'], \
                 type_converters=weave.converters.blitz, compiler='gcc')
    return indices

def printXtructure(self):
    '''
    Print out a summary of how many X are n-fold in the particle system.
    '''
    s = set(self.ps['id'])
    l = [ self.ps['id'].count(str) for str in s ]
    k = [ l.count(i) * i for i in range(1, 101) ]
    print k, sum(k)

def estimateProposalDistr(self):
    '''
    Estimate the parameters of the proposal distribution from the particle system.
    '''

    # aggregate particle weights for faster estimation

    weights = self.normalize()
    fdata = []; fweights = []
    sorted = argsort(self.ps['id'])
    particle = X[sorted[0]]
    weight = weights[sorted[0]]; count = 1
    for index in sorted[1:]:
        if (particle == X[index]).all():
            count += 1
        else:
            fdata.append(particle)
            fweights.append(weight * count)
            particle = X[index]
            weight = weights[index]
            count = 1
