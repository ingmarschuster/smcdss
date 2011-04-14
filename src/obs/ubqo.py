#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Unconstrained Binary Quadratic Optimization.
"""

"""
@namespace obs.ubqo
$Author$
$Rev$
$Date$
@details
"""

import os
import obs
import numpy
import scipy.stats as stats
import utils
import cPickle as pickle

class ubqo():
    def __init__(self, v, problem=None):
        if problem is None: problem = v['RUN_PROBLEM'][0]
        self.problem = problem
        self.A = load_ubqo_problem(v['RUN_TESTSUITE'])[problem - 1]['problem']
        self.best_obj = load_ubqo_problem(v['RUN_TESTSUITE'])[problem - 1]['best_obj']
        self.d = self.A.shape[0]
        self.v = v

def import_beasly_lib(filename):
    """
        Import problems from http://people.brunel.ac.uk/~mastjjb/jeb/orlib/bqpinfo.html used in
        Heuristic algorithms for the unconstrained binary quadratic programming,
        J.E. Beasley 1998
    """
    path = os.path.join(obs.v['SYS_ROOT'], 'data', 'ubqp')
    file = open(os.path.join(path, 'beasly', filename + '.txt'), 'r')
    L = list()
    n = int(file.readline())
    line = file.readline().strip().split(' ')

    best_obj = dict(bqp50=[2098, 3702, 4626, 3544, 4012, 3693, 4520, 4216, 3780, 3507],
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
        for i in xrange(A.shape[0]):
            for j in xrange(i):
                A[j, i] = A[i, j]
        L.append({'best_obj' : float(best_obj[filename][k]), 'problem' : A})

    file.close()
    return L

def import_glover_lib(filename):
    """
        Import problems from http://hces.bus.olemiss.edu/tools.html used in
        One-Pass Heuristics for Unconstrained Binary Quadratic Problems,
        F. Glover, B. Alidaee, C. Rego, and G. Kochenberger 2002
    """
    path = os.path.join(obs.v['SYS_ROOT'], 'data', 'ubqp')
    L = list()
    for k in xrange(5):
        file = open(os.path.join(path, 'glover', filename + chr(97 + k) + '.dat'), 'r')
        best_obj = float(file.readline().split(' = ')[1][:-3])
        d = int(file.readline().strip().split(' ')[0])
        print '%s\t d=%d' % (chr(97 + k), d)
        a = file.read()
        a = a.split('\n')
        while not utils.format.isnumeric(a[-1].strip().split(' ')[0]):
            a = a[:-1]
        a = ''.join(a)
        a = a.replace('\n', '').replace('\r', '')
        a = numpy.array([int(x) for x in a.split(' ') if not x == ''])
        print a.shape
        A = -a.reshape((d, d))
        L.append({'best_obj' : best_obj, 'problem' : A})
        file.close()
    return L

def random_cho(d):
    K = numpy.random.normal(size=(d, d))
    K = numpy.dot(K, K.T) + 1e-3 * numpy.eye(d)
    v = numpy.sqrt(K.diagonal()[numpy.newaxis, :])
    K /= numpy.dot(v.T, v)
    return numpy.linalg.cholesky(K)

def uniform(d, rho=1.0, xi=0.0, c=50):
    v = numpy.random.randint(low= -c / 2, high=c / 2, size=d) * (numpy.random.random(size=d) <= rho)
    return v + (v.max(), -v.min())[xi > 0] * xi

def student(d, rho=1.0, xi=0.0, c=50):
    v = numpy.array([stats.t._rvs(1) for i in xrange(d)]).T * c // 1 * (numpy.random.random(size=d) <= rho)
    return v + (v.max(), -v.min())[xi > 0] * xi

def normal(d, rho=1.0, xi=0.0, c=50):
    v = numpy.random.normal(size=d) * c // 1 * (numpy.random.random(size=d) <= rho)
    return v + (v.max(), -v.min())[xi > 0] * xi

def normal_mixture(d, rho=1.0, xi=0.0, c=50):
    p = 0.1
    v = numpy.array([numpy.random.normal()*(c, 10 * c)[numpy.random.random() < p] for i in xrange(d)]).T // 1 * (numpy.random.random(size=d) <= rho)
    return v + (v.max(), -v.min())[xi > 0] * xi

def generate_ubqo_problem(d, rho=1.0, xi=0.0, c=50, n=1, random=student, filename=None):
    """
        Generates a random UBQO problem.
    """
    path = os.path.join(obs.v['SYS_ROOT'], 'data', 'ubqp')
    L = list()
    for k in xrange(n):
        print '%s\t d=%d' % (k + 1, d)
        A = numpy.zeros((d, d))
        for i in xrange(d):
            A[i, : i + 1] = random(d=i + 1, rho=rho, xi=xi, c=c)
            A[:i, i] = A[i, :i]
        L.append({'best_obj' :1e-99, 'problem' : A})
        if not filename is None:
            file = open(os.path.join(path, filename + '.pickle'), 'w')
            pickle.dump(obj=L, file=file)
            file.close()
    return L

def pickle_ubqo_problem(filename):
    """
        Reads a UQBO problem from file and save it as pickled object.
    """
    path = os.path.join(obs.v['SYS_ROOT'], 'data', 'ubqp')
    file = open(os.path.join(path, filename + '.pickle'), 'w')
    if filename[:3] == 'bqp':
        pickle.dump(obj=import_beasly_lib(filename), file=file)
    else:
        pickle.dump(obj=import_glover_lib(filename), file=file)
    file.close()

def load_ubqo_problem(filename, repickle=False):
    """
        Loads a pickled UQBO problem.
    """
    path = os.path.join(obs.v['SYS_ROOT'], 'data', 'ubqp')
    path = os.path.join(path, filename + '.pickle')
    if repickle or not os.path.isfile(path): pickle_ubqo_problem(filename)
    file = open(path, 'r')
    L = pickle.load(file)
    file.close()
    for p in L:
        if p['best_obj'] <= 1e-99: p['best_obj'] = -numpy.inf
    return L

def main():
    x = generate_ubqo_problem(d=250, rho=1, xi=0, c=50, random=normal_mixture, filename='r250nm')
    print utils.format.format(x[0]['problem'])

if __name__ == "__main__":
    main()
