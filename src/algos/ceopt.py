#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from auxpy.data import data
from datetime import time
from numpy import zeros

from binary import *

def ceopt(target, verbose=False):
    '''
        Runs cross-entropy optimization.
        
        @param target target function
        @param verbose verbose
    '''

    start = clock()
    model = HybridBinary.uniform(target.d, model=dicCE['model'])
    print "running ceopt using " + model.name

    d = data()
    max = dict(state=None, score= -inf)

    # run optimization scheme
    for step in range(1, 100):
        if verbose: print "\nstep %i" % step,

        d.sample(model, dicCE['n_particles'], verbose=verbose)
        d.assign_weights(f=target, verbose=verbose)

        model.renew_from_data(sample=d.fraction(dicCE['elite']), verbose=verbose, **dicCE)
        if verbose: print "state: ", model.iZeros, model.iOnes, model.iModel

        # check if dimension is sufficiently reduced
        if model.nModel < 15:
            max = brute_search(target=target, max=dict(state=d._X[0], score=d._W[0]), model=model)
            stdout.write('\rscore: %.5f' % max['score'])
            return max, clock() - start

        max = dict(state=d._X[0], score=d._W[0])
        d.clear(fraction=dicCE['elite'])

        dicCE['elite'] *= 0.95

        stdout.write('\r%02i. %.5f [0: %03i, 1: %03i, r: %03i]' % (step, max['score'], model.nZeros, model.nOnes, model.nModel))
        stdout.flush()

    return max, clock() - start

def brute_search(target, max, model):
    '''
        Run exhaustive search.
        
        @param target target function
        @param max current max state and score 
        @param model underlying model
        @return max max state and score obtained from solving the sub-problem 
    '''
    if model.nModel > 0:
        gamma = model._Const
        for dec in range(2 ** model.nModel):
            bin = dec2bin(dec, model.nModel)
            gamma[model.iModel] = bin
            score = target.lpmf(gamma)
            if score > max['score']:
                max['state'] = gamma.copy()
                max['score'] = score
    return max
