#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

import sys
import obs
from binary import *

def solve_ce(f, verbose=False):
    '''
        Runs cross-entropy optimization.
        
        @param f f function
        @param verbose verbose
    '''

    t = time.time()
    model = obs.CE_BINARY_MODEL
    model = model.uniform(f.d)
    print "running ceopt using " + model.name

    d = utils.data.data()
    best_obj = -numpy.inf
    best_soln = numpy.zeros(f.d)

    # run optimization scheme
    for step in range(1, 100):
        if verbose: print "\nstep %i" % step,

        d.sample(model, obs.CE_N_PARTICLES, verbose=verbose)
        best_obj, best_soln = d.dichotomize_weights(f=f, fraction=obs.CE_ELITE)

        model.renew_from_data(sample=d, lag=obs.CE_LAG, verbose=verbose)

        # check if dimension is sufficiently reduced
        if len(model.r) < 12:
            best_obj, best_soln = obs.brute_force.solve_bf(f=f, best_obj=best_obj, \
                                                       gamma=best_soln, index=model.r)
            sys.stdout.write('\rscore: %.5f' % best_obj)
            return best_obj, best_soln, time.time() - t

        d.clear(fraction=obs.CE_ELITE)

        sys.stdout.write('\r%02i. objective: %.1f random: %i' % (step, best_obj, len(model.r)))
        sys.stdout.flush()

    return best_obj, best_soln, time.time() - t

def main():
    f = QuExpBinary(obs.ubqo.generate_ubqo_problem(35, p=1, c=50, n=1)[0][1])
    solve_ce(f)
    print
    print obs.solve_scip(f)

if __name__ == "__main__":
    main()
