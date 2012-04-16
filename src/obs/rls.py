#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Randomized 1-opt local search. \namespace obs.rls """

import time
import utils
import numpy
import binary.quadratic_exponential
import ubqo

class rls(ubqo.ubqo):
    name = 'RLS'
    header = []
    def run(self):
        return solve_rls(f=binary.quadratic_exponential.QuExpBinary(self.A), n=self.v['RLS_MAX_ITER'], m=self.v['RLS_MAX_TIME'])

def solve_rls(f, n=numpy.inf, m=numpy.inf, verbose=True):
    """
        Run simulated annealing optimization.
        @param f f function
        @param n number of steps
        @param m maximum time in minutes
        @param verbose verbose
    """

    print 'Running random local search...',
    if n < numpy.inf: print 'for %.f steps' % n
    if m < numpy.inf: print 'for %.2f minutes' % m

    t = time.time()
    k, s, = 0, 0
    best_soln = curr_soln = numpy.random.random(f.d) > 0.5
    best_obj = curr_obj = f.lpmf(curr_soln)
    next_obj = numpy.empty(f.d, dtype=float)

    while True:

        k += 1

        # update break criterion
        if n is numpy.inf: r = (time.time() - t) / (60.0 * m)
        else: r = k / float(n)

        # show progress bar
        if verbose:
            if r - s >= 0.01:
                utils.auxi.progress(r, ' obj: %.1f, time %s' % (best_obj, utils.auxi.time(time.time() - t)))
                s = r
        if r >= 1.0:
            utils.auxi.progress(1.0, ' obj: %.1f, time %s' % (best_obj, utils.auxi.time(time.time() - t)))
            break

        # restart algorithm
        if curr_obj > best_obj:
            best_obj = curr_obj
            best_soln = curr_soln

        curr_soln = numpy.random.random(f.d) > 0.5
        curr_obj = f.lpmf(curr_soln)

        # local search
        l_opt = False
        while not l_opt:
            k += 1
            curr_obj, l_opt = improve_soln(f, curr_soln, curr_obj, next_obj)

    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def improve_soln(f, curr_soln, curr_obj, next_obj):
    '''
        Get pivot index.
        \return index if solution can be improved
    '''
    for i in xrange(f.d):
        next_obj[i] = curr_obj + (f.A[i, i] + numpy.dot(f.A[i, :i], curr_soln[:i])
                                   + numpy.dot(f.A[i + 1:, i], curr_soln[i + 1:])) * (1 - 2 * curr_soln[i])
    if (next_obj <= curr_obj).all():
        return curr_obj, True
    else:
        i = numpy.argmax(next_obj)
        curr_soln[i] = 1 - curr_soln[i]
        return next_obj[i], False

    return next_obj, True

def get_permutation(d):
    """
        Draw a random permutation of the index set.
        \return permutation
    """
    perm = range(d)
    for i in reversed(range(1, d)):
        # pick an element in p[:i+1] with which to exchange p[i]
        j = numpy.random.randint(low=0, high=i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm
