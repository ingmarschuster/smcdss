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
        Run cross entropy optimization.
    '''

    n = dicCE['n_particles']
    d = target.d

    # presampling step
    start = clock()

    model = (MixtureBinary(
                 dHybrid=HybridBinary.uniform(d, model=dicCE['dep_model']),
                 rProd=dicCE['r_prod'],
                 lagProd=dicCE['lag_prod'],
                 lagHybrid=dicCE['lag_dep']
                 ))

    if verbose: print "start ceopt using " + model.dHybridCurrent.dDep.name

    d = data()
    #d.sample(model, n, verbose=verbose)
    #d.clear(fraction=dicCE['elite_prod'])

    #if verbose: print "score: %.5f\n" % d.getWeights(noexp=True)[0]

    # run crosss entropy optimization scheme

    for step in range(1, 51):
        if verbose: print "step %i" % step,

        # create weighted sample
        d.sample(model, n, verbose=verbose)
        d.assign_weights(f=target)

        # reinit sampler with elite samples
        model.renew_from_data(d, fProd=dicCE['elite_prod'], fHybrid=dicCE['elite_dep'], eps=dicCE['eps'])

        if verbose:
            x = model.dHybridCurrent
            print "state: " , x.iOnes, x.iZeros, x.iRand

        # if dimension is reduced to feasible size, run brute force search
        if model.dHybridCurrent.nRand < 5:
            state_max = d._X[0]
            score_max = d._w[0]
            if model.dHybridCurrent.nRand > 0:
                gamma = model.dHybridCurrent._cBase
                for dec in range(2 ** model.dHybridCurrent.nRand):
                    bin = dec2bin(dec, model.dHybridCurrent.nRand)
                    gamma[model.dHybridCurrent.iRand] = bin
                    score = target.lpmf(gamma)
                    if score > score_max:
                        state_max = gamma.copy()
                        score_max = score

            return ['%.3f' % score_max, '[' + ', '.join([str(i) for i in where(state_max)[0]]) + ']',
                    '%.3f' % (clock() - start)]

        # remove all but elite samples
        d.clear(fraction=dicCE['elite_prod'])

        state_max = d._X[0]
        score_max = d._w[0]

        if verbose: print "score: %.5f\n" % d.getWeights(noexp=True)[0]

target = PosteriorBinary(dataFile='/home/cschafer/Documents/smcdss/data/datasets/test_dat.csv')
print ceopt(target)
