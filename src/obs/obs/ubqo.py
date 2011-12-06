#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Unconstrained Binary Quadratic Optimization.

@verbatim
USAGE:
        ubqo <option>

OPTIONS:
        -d    dimension (integer)
        -t    type (text, 'uniform','cauchy','normal')
        -f    filename
        -c    completeness; expected ratio of non-zeros over all entries
        -s    skewness (float on (-1,-1)); the values will come from [[-range(1+s),range(1+s)]]
        -r    range (integer); the values will come from [[-range,range]]

@endverbatim
"""

"""
@namespace obs.ubqo
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (lun., 10 oct. 2011) $
@details
"""

import getopt
import sys
import os
import numpy
import scipy.stats as stats
import utils
import cPickle as pickle
import obs
from numpy import inf

class ubqo():
    def __init__(self, v):
        
        # problem number
        self.problem = v['RUN_PROBLEM']
        self.testsuite = v['RUN_TESTSUITE']
        
        # read specific problem
        self.A, self.primal_bound, self.dual_bound = read_problem(self.testsuite, self.problem)
        
        # save parameters
        self.d, self.v = self.A.shape[0], v

def write_problem(testsuite, problem, A, primal_bound, dual_bound):
    """
        Write a UQBO problem to file.
    """
    # write matrix
    d = A.shape[0]
    
    # find path
    path = os.path.join(obs.v['DATA_PATH'], testsuite)
    if not os.path.isdir(path): os.mkdir(path)
    path = os.path.join(path, testsuite + '_%02d.dat' % problem)
    
    file = open(path, 'w')
    file.write('%.f\n' % primal_bound)    
    file.write('%.10f\n' % dual_bound)
    file.write('%d\n' % d)
    for j in xrange(d):
        for i in xrange(j, d):
            if i == j: a = A[i, j]
            else: a = 2 * A[i, j]
            file.write('%d\n' % a)
    file.close()
    
def read_problem(testsuite, problem):
    """
        Reads a UQBO problem from file.
    """
    path = os.path.join(obs.v['DATA_PATH'], testsuite, testsuite + '_%02d.dat' % problem)

    # read matrix
    file = open(path, 'r')  
    primal_bound, dual_bound, d = eval(file.readline()), eval(file.readline()), int(file.readline())
    A = numpy.zeros(shape=(d, d))
    for j in xrange(d):
        for i in xrange(j, d):
            A[i, j] = float(file.readline())
    file.close()
    return A, primal_bound, dual_bound

def uniform(d, rho=1.0, xi=0.0, c=50):
    v = numpy.random.randint(low= -c, high=c, size=d) * (numpy.random.random(size=d) <= rho)
    return v + (v.max(), -v.min())[xi > 0] * xi

def cauchy(d, rho=1.0, xi=0.0, c=50):
    v = numpy.array([stats.cauchy.rvs(1) for i in xrange(d)]).T * c // 1 * (numpy.random.random(size=d) <= rho)
    return v + (v.max(), -v.min())[xi > 0] * xi

def normal(d, rho=1.0, xi=0.0, c=50):
    v = numpy.random.normal(size=d) * c // 1 * (numpy.random.random(size=d) <= rho)
    return v + (v.max(), -v.min())[xi > 0] * xi

def normal_mixture(d, rho=1.0, xi=0.0, c=50):
    p = 0.1
    v = numpy.array([numpy.random.normal()*(c, 10 * c)[numpy.random.random() < p] for i in xrange(d)]).T // 1 * (numpy.random.random(size=d) <= rho)
    return v + (v.max(), -v.min())[xi > 0] * xi

def generate_ubqo_problem(d, completeness=1.0, skewness=0.0, range=50, n=1, type=cauchy, testsuite=None):
    """
        Generates a random UBQO problem.
    """
    print "Generate %d problem(s) of dimension %d of type '%s' on the range [[-%d,%d]]." % (n, d, type.__name__, range, range)
    for problem in xrange(1, n + 1):
        A = numpy.zeros((d, d))
        for i in xrange(d):
            A[i, : i + 1] = type(d=i + 1, rho=completeness, xi=skewness, c=range)
            A[:i, i] = A[i, :i]
        write_problem(testsuite, problem, A, -numpy.inf, numpy.inf)

def main():
    # Parse command line options.
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:t:f:c:s:r:n:')
    except getopt.error, msg:
        print msg
        sys.exit(2)
        
    d, type, testsuite = None, None, None
    n, completeness, skewness, range = 1, 1.0, 0.0, 100
    
    for o, a in opts:
        if o == '-d': d = int(a)
        if o == '-n': n = int(a)
        if o == '-t': type = eval(a)
        if o == '-f': testsuite = a
        if o == '-c': completeness = float(a)
        if o == '-s': skewness = float(a)
        if o == '-r': range = int(a)
        if o == '-n': n = int(a)
    
    for var in [d, type, testsuite]:
        if var is None:
            print "You need to specify dimension, type and testsuite."
            sys.exit(0)
    
    obs.read_config()
    generate_ubqo_problem(d=d, completeness=completeness, skewness=skewness, n=n, range=range, type=type, testsuite=testsuite)
    print "Test suite saved as '%s'." % testsuite

if __name__ == "__main__":
    main()
    


    
#------------------------------------------------------------------------------ 




def import_beasly_lib():
    """
        Import problems from http://people.brunel.ac.uk/~mastjjb/jeb/orlib/bqpinfo.html used in
        Heuristic algorithms for the unconstrained binary quadratic programming,
        J.E. Beasley 1998
    """
    primal_bound = dict(bqp50=[2098, 3702, 4626, 3544, 4012, 3693, 4520, 4216, 3780, 3507],
                    bqp100=[7970, 11036, 12723, 10368, 9083, 10210, 10125, 11435, 11435, 12565],
                    bqp250=[45607, 44810, 49037, 41274, 47961, 41014, 46757, 35726, 48916, 40442],
                    bqp500=[116586, 128223, 130812, 130097, 125487, 121719, 122201, 123559, 120798, 130619],
                    bqp1000=[371438, 354932, 371226, 370560, 352736, 359452, 370999, 351836, 348732, 351415],
                    bqp2500=[1515011, 1468850, 1413083, 1506943, 1491796, 1468427, 1478654, 1484199, 1482306, 1482354])

    for filename in primal_bound.keys():
        if filename in ['bqp1000', 'bqp2500']: continue
    
        file = open(os.path.join(obs.v['DATA_PATH'], 'archive', 'beasly', filename + '.txt'), 'r')
        n = int(file.readline())
        line = file.readline().strip().split(' ')
    
        for problem in xrange(n):
            print '%d\t d=%s, nonzeros=%s' % (problem + 1, line[0], line[1])
            d = int(line[0])
            A = numpy.zeros((d, d))
            line = file.readline().strip().split(' ')
            while len(line) == 3:
                A[int(line[1]) - 1, int(line[0]) - 1] = float(line[2])
                line = file.readline().strip().split(' ')
            for i in xrange(A.shape[0]):
                for j in xrange(i):
                    A[j, i] = A[i, j]
            
            write_problem(filename, problem + 1, A, float(primal_bound[filename][problem]), numpy.inf)
    
        file.close()

def import_glover_lib():
    """
        Import problems from http://hces.bus.olemiss.edu/tools.html used in
        One-Pass Heuristics for Unconstrained Binary Quadratic Problems,
        F. Glover, B. Alidaee, C. Rego, and G. Kochenberger 2002
    """
    path = os.path.join(obs.v['DATA_PATH'], 'archive', 'glover')
    for filename in ['f1', 'f2', 'g2']:
        for problem in xrange(5):
            file = open(os.path.join(path, filename + chr(97 + problem) + '.dat'), 'r')
            primal_bound = float(file.readline().split(' = ')[1][:-3])
            d = int(file.readline().strip().split(' ')[0])
            print '%s\t d=%d' % (filename + chr(97 + problem), d)
            a = file.read()
            a = a.split('\n')
            while not utils.aux.isnumeric(a[-1].strip().split(' ')[0]):
                a = a[:-1]
            a = ''.join(a)
            a = a.replace('\n', '').replace('\r', '')
            a = numpy.array([int(x) for x in a.split(' ') if not x == ''])
            A = -a.reshape((d, d))
            write_problem(filename, problem + 1, A, primal_bound, numpy.inf)
            file.close()
