#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-10-29 20:13:19 +0200 (ven., 29 oct. 2010) $
    $Revision: 30 $
'''

__version__ = "$Revision: 30 $"

from auxpy.data import data
from datetime import time
from numpy import zeros

from binary import *
from auxpy.default import dicCE, dicData

def ceopt(target, verbose=True):
    '''
        Runs cross-entropy optimization.
        
        @param target target function
    '''

    start = clock()

    model = MixtureBinary(dHybrid=HybridBinary.uniform(target.d, model=dicCE['dep_model']), lag=dicCE['lag'])
    if verbose: print "start ceopt using " + model.dHybridCurrent.dDep.name
    d = data()

    # run optimization scheme
    for step in range(1, 50):
        if verbose: print "step %i" % step,

        d.sample(model, dicCE['n_particles'], verbose=verbose)
        d.assign_weights(f=target)
        
        model.renew_from_data(d, fProd=dicCE['elite_prod'], fDep=dicCE['elite_dep'], eps=dicCE['eps'], verbose=verbose)
        if verbose: print "state: ", model.dHybridCurrent.iOnes, model.dHybridCurrent.iZeros, model.dHybridCurrent.iRand

        # check if dimension is sufficiently reduced
        if model.dHybridCurrent.nRand < 5:
            max = brute_search(target=target, max=dict(state=d._X[0], score=d._w[0]), model=model)
            return max, clock() - start

        d.clear(fraction=dicCE['elite_prod'])

        max = dict(state=d._X[0], score=d._w[0])
        if verbose: print "score: %.5f\n" % max['score']

def brute_search(target, max, model):
    '''
        Run exhaustive search.
        
        @param target target function
        @param max current max state and score 
        @param model underlying model
        @return max max state and score obtained from solving the sub-problem 
    '''
    if model.dHybridCurrent.nRand > 0:
        gamma = model.dHybridCurrent._cBase
        for dec in range(2 ** model.dHybridCurrent.nRand):
            bin = dec2bin(dec, model.dHybridCurrent.nRand)
            gamma[model.dHybridCurrent.iRand] = bin
            score = target.lpmf(gamma)
            if score > max['score']:
                max['state'] = gamma.copy()
                max['score'] = score
    return max

target = PosteriorBinary(dataFile='/home/cschafer/Documents/smcdss/data/datasets/test_dat.csv')
max, time= ceopt(target)
print max['score']
print '[' + ', '.join([str(i) for i in where(max['state'])[0]]) + ']',
print time
