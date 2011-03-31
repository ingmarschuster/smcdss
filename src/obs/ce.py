#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#    $Author: Christian Schäfer
#    $Date$

__version__ = "$Revision$"

import sys
import obs
from binary import *

def solve_ce(f, verbose=True):
    ''' Runs cross-entropy optimization.
        @param f f function
        @param verbose verbose
    '''

    t = time.time()
    model = obs.CE_BINARY_MODEL
    model = model.uniform(f.d)
    print 'running ceopt using ' + model.name

    d = utils.data.data()
    best_obj = -numpy.inf
    best_soln = numpy.zeros(f.d)

    # run optimization scheme
    for step in xrange(1, 100):

        d.sample(model, obs.CE_N_PARTICLES, verbose=False)
        best_obj, best_soln = d.dichotomize_weights(f=f, fraction=obs.CE_ELITE)

        model.renew_from_data(sample=d, lag=obs.CE_LAG, verbose=False)

        # check if dimension is sufficiently reduced
        if len(model.r) < 12:
            best_obj, best_soln = \
                obs.solve_bf(f=f, best_obj=best_obj, gamma=best_soln, index=model.r)
            break

        d.clear(fraction=obs.CE_ELITE)

        if verbose:
            sys.stdout.write('\r%02i.\tobjective: %.1f\trandom: %i' % (step, best_obj, len(model.r)))
            sys.stdout.flush()

    if verbose: sys.stdout.write('\n')
    return best_obj, best_soln, 'ce', time.time() - t

def main():
    suite= obs.ubqo.load_ubqo_problem(filename='f2')
    f = QuExpBinary(suite[3][1])
    print suite[3][0]
    print obs.solve_ce(f)
    print obs.solve_sa(f, 1e5)
    f = QuExpBinary(obs.ubqo.generate_ubqo_problem(30, p=0.95, c=50, n=1)[0][1])
    print
    print solve_sa(f, n=1e5)
    #print obs.solve_ce(f)
    #print obs.solve_scip(f)

if __name__ == "__main__":
    main()
