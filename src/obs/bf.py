#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#    $Author: Christian Schäfer
#    $Date: 2011-03-07 17:03:12 +0100 (lun., 07 mars 2011) $

__version__ = "$Revision: 91 $"

import numpy
import utils

def solve_bf(f, best_obj= -numpy.inf, gamma=None, index=None):
    '''
        Finds a maximum via exhaustive enumeration.
        @param f f function
        @param best_obj current best objective
        @param gamma binary vector of super problem dimension
        @param index index list indicating subproblem
        @return best_obj best objective after solving the sub-problem
        @return best_soln best solution after solving the sub-problem
        @todo Write cython version of brute force search.
    '''
    if not index is None:
        d = len(index)
    else:
        d = f.d
        gamma = numpy.zeros(d)
    best_soln = gamma.copy()
    if d > 0:
        for dec in range(2 ** d):
            bin = utils.format.dec2bin(dec, d)
            gamma[index] = bin
            v = f.lpmf(gamma)
            if v > best_obj:
                best_obj = v
                best_soln = gamma.copy()
    return best_obj, best_soln

def main():
    pass

if __name__ == "__main__":
    main()