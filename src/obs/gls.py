#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Ramdomized local search heuristic (Boros et al. 2006). \namespace obs.gls """

import time
import utils
import numpy
import binary.quadratic_exponential
import ubqo

class gls(ubqo.ubqo):
    name = 'GLS'
    header = []
    def run(self):
        return solve_gls(f=binary.quadratic_exponential.QuExpBinary(self.A), n=self.v['GLS_MAX_ITER'], m=self.v['GLS_MAX_TIME'])

def solve_gls(f, n=numpy.inf, m=numpy.inf, verbose=True):
    """
        Run simulated annealing optimization.
        @param f f function
        @param n number of steps
        @param m maximum time in minutes
        @param verbose verbose
    """

    print 'Running gradient local search...',
    if n < numpy.inf: print 'for %.f steps' % n
    if m < numpy.inf: print 'for %.2f minutes' % m

    t = time.time()
    k, s, = 0, 0
    best_soln = curr_soln = numpy.random.random(f.d)
    best_obj = f.lpmf(curr_soln)
    delta = get_derivative(f, curr_soln)

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

        # get pivot element        
        i = get_pivot(f, curr_soln, delta)

        # restart algorithm
        if i is None:
            curr_obj = f.lpmf(curr_soln)
            if curr_obj > best_obj:
                best_obj = curr_obj
                best_soln = curr_soln
            curr_soln = numpy.random.random(f.d)
            delta = get_derivative(f, curr_soln)
            continue

        # improve solution
        improve_soln(curr_soln, i, delta, f)

    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def get_derivative(f, x):
    '''
        Compute the derivative of the target function.
    '''
    delta = numpy.zeros_like(x)
    for i in xrange(x.shape[0]):
        delta[i] += f.A[i, i]
        for j in xrange(i): delta[i] += f.A[i, j] * x[j]
        for j in xrange(i + 1, x.shape[0]): delta[i] += f.A[j, i] * x[j]
    return delta

def get_pivot(f, p, delta):
    '''
        Get pivot index.
        \return index if solution can be improved
    '''
    dist = numpy.maximum(-p * delta, (1 - p) * delta)
    i = numpy.argmax(dist)
    if not dist[i] > 0: return
    else: return i

def improve_soln(x, i, delta, f):
    '''
        Improve the solution and adjust the derivative.
    '''
    p = x[i]
    x[i] = [0, 1][delta[i] >= 0]
    for j in xrange(i): delta[j] += f.A[i, j] * (x[i] - p)
    for j in xrange(i + 1, x.shape[0]): delta[j] += f.A[j, i] * (x[i] - p)

