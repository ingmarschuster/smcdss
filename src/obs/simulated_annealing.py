#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2011-03-29 16:06:25 +0200 (mar., 29 mars 2011) $
    $Revision: 94 $
'''

__version__ = "$Revision: 94 $"

import sys
from binary import *

def solve_sa(f, n, verbose=False):
    '''
        Run simulated annealing optimization.
    '''

    t = time.time()
    best_soln = None
    best_obj = -numpy.inf
    curr_soln = ProductBinary.uniform(d=f.d).rvs()
    curr_obj = f.lpmf(curr_soln)

    sys.stdout.write("\n" + 101 * " " + "]" + "\r" + "["); progress = 0
    for t in range(1, n + 1):

        progress_next = 100 * t / n
        if 100 * n % t == 0:
            sys.stdout.write((progress_next - progress) * "-")
            sys.stdout.flush()
            progress = progress_next

        # generate proposal
        proposal = curr_soln.copy()
        index = numpy.random.randint(0, f.p)
        proposal[index] = proposal[index] ^ True
        score_proposal = f.lpmf(proposal)

        if best_obj < score_proposal:
            best_obj = score_proposal
            best_soln = proposal.copy()

        if (score_proposal - curr_obj) * t >  numpy.log(numpy.random.random()):
            curr_soln = proposal
            curr_obj = score_proposal

    best_soln = numpy.where(numpy.array(best_soln) == True)[0]

    return best_obj, best_soln, t - time.time()


