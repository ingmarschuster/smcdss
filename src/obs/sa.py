#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#    $Author: Christian Schäfer
#    $Date: 2011-03-07 17:03:12 +0100 (lun., 07 mars 2011) $

__version__ = "$Revision: 94 $"

import sys
import obs
from binary import *

def solve_sa(f, n, verbose=True):
    ''' Run simulated annealing optimization.
        @param f f function
        @param n number of steps
        @param verbose verbose
    '''

    print 'running simulated annealing for %i steps' % n

    t = time.time()
    n = int(n)
    best_soln = None
    best_obj = -numpy.inf
    curr_soln = ProductBinary.uniform(d=f.d).rvs()
    curr_obj = f.lpmf(curr_soln)

    for k in xrange(1, n + 1):

        if verbose: utils.format.progress(k, n, ' objective: %.1f' % best_obj)

        # generate proposal
        proposal = curr_soln.copy()
        index = numpy.random.randint(0, f.d)
        proposal[index] = proposal[index] ^ True
        proposal_obj = f.lpmf(proposal)

        if best_obj < proposal_obj:
            best_obj = proposal_obj
            best_soln = proposal.copy()

        if (proposal_obj - curr_obj) * k / float(n) > numpy.log(numpy.random.random()):
            curr_soln = proposal
            curr_obj = proposal_obj

    if verbose: sys.stdout.write('\n')
    return best_obj, best_soln, 'sa', time.time() - t

def main():
    pass

if __name__ == "__main__":
    main()
