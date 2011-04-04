#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#    $Author: Christian Schäfer
#    $Date: 2011-03-07 17:03:12 +0100 (lun., 07 mars 2011) $

__version__ = "$Revision: 94 $"

from obs import *

class sa(ubqo.ubqo):
    header = ['NO_MOVES', 'ACC_RATE']
    def run(self):
        return solve_sa(f=binary.QuExpBinary(self.A), n=self.v['SA_MAX_ITER'], m=self.v['SA_MAX_TIME'])

def solve_sa(f, n=numpy.inf, m=numpy.inf, verbose=True):
    ''' Run simulated annealing optimization.
        @param f f function
        @param n number of steps
        @param m maximum time in minutes
        @param verbose verbose
    '''

    print 'running simulated annealing',
    if n < numpy.inf: print 'for %.f steps' % n
    if m < numpy.inf: print 'for %.2f minutes' % m

    t = time.time()
    k, s = 0, 0
    best_soln = None
    best_obj = -numpy.inf
    curr_soln = binary.ProductBinary.uniform(d=f.d).rvs()
    curr_obj = f.lpmf(curr_soln)

    while True:

        # update break criterion
        if n is numpy.inf:
            r = (time.time() - t) / (60.0 * m)
        else:
            k += 1
            r = k / n

        # show progress bar
        if verbose:
            if r - s >= 0.01:
                utils.format.progress(r, ' objective: %.1f' % best_obj)
                s = r

        if r > 1:
            if verbose: utils.format.progress(1.0, ' objective: %.1f' % best_obj)
            break

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
    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def main():
    pass

if __name__ == "__main__":
    main()
