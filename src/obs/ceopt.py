#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

import sys

from obs import *

def ceopt(f, verbose=False):
    '''
        Runs cross-entropy optimization.
        
        @param f f function
        @param verbose verbose
    '''

    t = time.time()
    model = hybrid_model.HybridBinary.uniform(f.d, model=CE_BINARY_MODEL)
    print "running ceopt using " + model.name

    d = utils.data.data()
    best_soln = dict(state=None, score= -numpy.inf)

    # run optimization scheme
    for step in range(1, 100):
        if verbose: print "\nstep %i" % step,

        d.sample(model, CE_N_PARTICLES, verbose=verbose)
        d.assign_weights(f=f, verbose=verbose)

        model.renew_from_data(sample=d.fraction(CE_ELITE), verbose=verbose)
        if verbose: print "state: ", model.iZeros, model.iOnes, model.iModel

        # check if dimension is sufficiently reduced
        if model.nModel < 15:
            best_soln = brute_force.solve_bf(f=f, best_soln=dict(state=d._X[0], score=d._W[0]), model=model)
            sys.stdout.write('\rscore: %.5f' % best_soln['score'])
            return best_soln, time.time() - t

        best_soln = dict(state=d._X[0], score=d._W[0])
        d.clear(fraction=CE_ELITE)

        sys.stdout.write('\r%02i. %.5f [0: %03i, 1: %03i, r: %03i]' % (step, best_soln['score'], model.nZeros, model.nOnes, model.nModel))
        sys.stdout.flush()

    return best_soln, time.time() - t
