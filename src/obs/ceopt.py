#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

import time
import sys
import numpy

import binary
import obs
import utils

def ceopt(target, verbose=False):
    '''
        Runs cross-entropy optimization.
        
        @param target target function
        @param verbose verbose
    '''

    t = time.time()
    model = binary.HybridBinary.uniform(target.d, model=obs.CE_BINARY_MODEL)
    print "running ceopt using " + model.name

    d = utils.data.data()
    max = dict(state=None, score= -numpy.inf)

    # run optimization scheme
    for step in range(1, 100):
        if verbose: print "\nstep %i" % step,

        d.sample(model, obs.CE_N_PARTICLES, verbose=verbose)
        d.assign_weights(f=target, verbose=verbose)

        model.renew_from_data(sample=d.fraction(obs.CE_ELITE), verbose=verbose)
        if verbose: print "state: ", model.iZeros, model.iOnes, model.iModel

        # check if dimension is sufficiently reduced
        if model.nModel < 15:
            max = brute_search(target=target, max=dict(state=d._X[0], score=d._W[0]), model=model)
            sys.stdout.write('\rscore: %.5f' % max['score'])
            return max, time.time() - t

        max = dict(state=d._X[0], score=d._W[0])
        d.clear(fraction=obs.CE_ELITE)

        sys.stdout.write('\r%02i. %.5f [0: %03i, 1: %03i, r: %03i]' % (step, max['score'], model.nZeros, model.nOnes, model.nModel))
        sys.stdout.flush()

    return max, time.time() - t

def brute_search(target, max, model):
    '''
        Run exhaustive search.
        
        @param target target function
        @param max current max state and score 
        @param model underlying model
        @return max max state and score obtained from solving the sub-problem
        @todo Write cython version of brute force search.
    '''
    if model.nModel > 0:
        gamma = model._Const
        for dec in range(2 ** model.nModel):
            bin = utils.format.dec2bin(dec, model.nModel)
            gamma[model.iModel] = bin
            score = target.lpmf(gamma)
            if score > max['score']:
                max['state'] = gamma.copy()
                max['score'] = score
    return max
