#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Brute force search.
"""

"""
@namespace obs.bf
$Author$
$Rev$
$Date$
@details
"""

import time
import utils
import numpy
import binary
import ubqo

class bf(ubqo.ubqo):
    header = []
    name = 'BF'
    def run(self):
        return solve_bf(f=binary.QuExpBinary(self.A))

def solve_bf(f, best_obj= -numpy.inf, gamma=None, index=None):
    """
        Finds a maximum via exhaustive enumeration.
        \param f f function
        \param best_obj current best objective
        \param gamma binary vector of super problem dimension
        \param index index list indicating subproblem
        \return best_obj best objective after solving the sub-problem
        \return best_soln best solution after solving the sub-problem
        @todo Write cython version of brute force search.
    """

    t = time.time()
    if not index is None:
        d = len(index)
    else:
        d = f.d
        gamma = numpy.zeros(d)
    best_soln = gamma.copy()
    if d > 0:
        for dec in xrange(2 ** d):
            bin = utils.aux.dec2bin(dec, d)
            gamma[index] = bin
            v = f.lpmf(gamma)
            if v > best_obj:
                best_obj = v
                best_soln = gamma.copy()
    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def main():
    pass

if __name__ == "__main__":
    main()
