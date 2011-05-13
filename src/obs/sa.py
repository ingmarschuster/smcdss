#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Simulated annealing.
"""

"""
@namespace obs.sa
$Author$
$Rev$
$Date$
@details
"""

from obs import *

class sa(ubqo.ubqo):
    name = 'SA'
    header = ['NO_MOVES', 'ACC_RATE']
    def run(self):
        return solve_sa(f=binary.QuExpBinary(self.A), n=self.v['SA_MAX_ITER'], m=self.v['SA_MAX_TIME'])

def solve_sa(f, n=numpy.inf, m=numpy.inf, verbose=True):
    """ Run simulated annealing optimization.
        @param f f function
        @param n number of steps
        @param m maximum time in minutes
        @param verbose verbose
    """

    print 'running simulated annealing',
    if n < numpy.inf: print 'for %.f steps' % n
    if m < numpy.inf: print 'for %.2f minutes' % m

    t = time.time()
    a, k, s, v = 0, 0, 0, 1e-10
    best_obj, best_soln = -numpy.inf, None
    curr_soln = binary.ProductBinary.uniform(d=f.d).rvs()
    curr_obj = f.lpmf(curr_soln)

    while True:

        k += 1

        # update break criterion
        if n is numpy.inf:
            r = (time.time() - t) / (60.0 * m)
        else:
            r = k / float(n)

        # show progress bar
        if verbose:
            if r - s >= 0.01:
                utils.format.progress(r, 'ar: %.3f, objective: %.1f, time %s' % (a / float(k), best_obj, utils.format.time(time.time() - t)))
                s = r

        if r > 1:
            if verbose: utils.format.progress(1.0, ' objective: %.1f, time %s' % (best_obj, utils.format.time(time.time() - t)))
            break

        # generate proposal
        proposal = curr_soln.copy()
        index = numpy.random.randint(0, f.d)
        proposal[index] = proposal[index] ^ True
        proposal_obj = f.lpmf(proposal)

        if best_obj < proposal_obj:
            best_obj = proposal_obj
            best_soln = proposal.copy()

        if (proposal_obj - curr_obj) * v > numpy.log(numpy.random.random()):
            a += 1
            curr_soln = proposal
            curr_obj = proposal_obj

        if a / float(k) < (r + 1) ** -5: v *= 0.995
        else:  v *= 1.005

    if verbose: sys.stdout.write('\n')
    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def main():
    pass

if __name__ == "__main__":
    main()
