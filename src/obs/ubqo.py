#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2011-02-17 18:34:45 +0100 (jeu., 17 févr. 2011) $
    $Revision: 73 $
'''

import os
import numpy
import cPickle as pickle

__PATH = '../../data/ubqp'

def import_beasly_lib(filename):
    '''
        Import problems from http://people.brunel.ac.uk/~mastjjb/jeb/orlib/bqpinfo.html used in
        Heuristic algorithms for the unconstrained binary quadratic programming,
        J.E. Beasley 1998
    '''
    file = open(os.path.join(__PATH, 'beasly', file + '.txt'), 'r')
    L = list()
    n = int(file.readline())
    line = file.readline().strip().split(' ')

    best_soln = dict(bqp50=[2098, 3702, 4626, 3544, 4012, 3693, 4520, 4216, 3780, 3507],
                    bqp100=[7970, 11036, 12723, 10368, 9083, 10210, 10125, 11435, 11435, 12565],
                    bqp250=[45607, 44810, 49037, 41274, 47961, 41014, 46757, 35726, 48916, 40442],
                    bqp500=[116586, 128223, 130812, 130097, 125487, 121719, 122201, 123559, 120798, 130619],
                    bqp1000=[371438, 354932, 371226, 370560, 352736, 359452, 370999, 351836, 348732, 351415],
                    bqp2500=[1515011, 1468850, 1413083, 1506943, 1491796, 1468427, 1478654, 1484199, 1482306, 1482354])

    for k in xrange(n):
        print '%d\t d=%s, nonzeros=%s' % (k + 1, line[0], line[1])
        d = int(line[0])
        A = numpy.zeros((d, d))
        line = file.readline().strip().split(' ')
        while len(line) == 3:
            A[int(line[1]) - 1, int(line[0]) - 1] = float(line[2])
            line = file.readline().strip().split(' ')
        L.append((float(best_soln[file][k]), A))

    file.close()
    return L

def import_glover_lib(filename):
    '''
        Import problems from http://hces.bus.olemiss.edu/tools.html used in
        One-Pass Heuristics for Unconstrained Binary Quadratic Problems,
        F. Glover, B. Alidaee, C. Rego, and G. Kochenberger 2002
    '''
    L = list()
    for k in xrange(5):
        file = open(os.path.join(__PATH, 'glover', filename + chr(97 + k) + '.dat'), 'r')
        best_soln = float(file.readline().split(' = ')[1][:-3])
        d = int(file.readline().strip().split(' ')[0])
        print '%s\t d=%d' % (chr(97 + k), d)
        a = file.read()
        a = a.split('\n')[:-3]
        a = ''.join(a)
        a = a.replace('\n', '').replace('\r', '')
        A = numpy.array([int(x) for x in a.split(' ') if not x == '']).reshape((d, d))
        L.append((best_soln, A))
        file.close()
    return L

def generate_ubqo_problem(d, p, c, n=1, filename=None):
    L = list()
    for k in xrange(n):
        print '%s\t d=%d' % (k + 1, d)
        A = numpy.zeros((d, d))
        for i in xrange(d):
            if p > numpy.random.random(): A[i, i] = numpy.random.randint(-c, c)
            for j in xrange(i):
                if p > numpy.random.random():
                    A[i, j] = numpy.random.randint(-c, c)
                    A[j, i] = A[i, j]
        L.append((None, A))
        if not filename is None:
            file = open(os.path.join(__PATH, filename + '.pickle'), 'w')
            pickle.dump(obj=L, file=file)
            file.close()
    return L

def pickle_ubqo_problem(filename):
    '''
        Reads a UQBO problem from file and save it as pickled object.
    '''
    file = open(os.path.join(__PATH, filename + '.pickle'), 'w')
    if filename[:3] == 'bqp':
        pickle.dump(obj=import_beasly_lib(filename), file=file)
    else:
        pickle.dump(obj=import_glover_lib(filename), file=file)
    file.close()

def load_ubqo_problem(filename, repickle=False):
    '''
        Loads a pickled UQBO problem.
    '''
    path = os.path.join(__PATH, file + '.pickle')
    if repickle or not os.path.isfile(path): pickle_ubqo_problem(filename)
    file = open(path, 'r')
    return pickle.load(file)
    file.close()
