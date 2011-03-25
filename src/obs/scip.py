#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2011-02-16 11:28:38 +0100 (mer., 16 févr. 2011) $
    $Revision: 71 $
'''

import numpy
import time
import os
import subprocess

import utils.format

def solve_scip(A):
    '''
        Solve UBQO using SCIP and ZIMPL from the ZIB Optimization Suite http://zibopt.zib.de.
        @param A lower triangle matrix
        @return objective maximum
        @return gamma maximizer
    '''
    
    # write matrix
    f = open('matrix.dat', 'w')
    f.write('%i\n' % A.shape[0])
    for j in xrange(A.shape[0]):
        for i in xrange(j, A.shape[0]):
            f.write('%.16f\n' % A[i, j])
    f.close()

    # invoke scip
    if os.path.exists('scip.log'): os.remove('scip.log')
    subprocess.Popen(['/home/cschafer/ziboptsuite-2.0.1/scip-2.0.1/bin/scip', '-l', 'scip.log', '-f', 'uqbo.zpl', '-q']).wait()

    # read log
    f = open('scip.log', 'r')
    s = f.read()
    f.close
    s = s.split('primal solution:\n================\n\n', 2)[1]
    s = s.split('\n\nStatistics\n==========\n', 2)[0]
    s = s.split('\n')

    # retrieve objective
    objective = float(s[0][16:].strip())

    # retrieve gamma
    gamma = numpy.zeros(A.shape[0], dtype=bool)
    for x in s[1:]:
        x = x.split()[0].split('#')[1:]
        if x[0] == x[1]: gamma[int(x[0]) - 1] = True

    return objective, gamma

def solve_bf(A):
    '''
        Solve UBQO using brute force.
        @param A lower triangle matrix
        @return objective maximum
        @return gamma maximizer
    '''
    objective = -numpy.inf
    gamma = numpy.zeros(A.shape[0], dtype=bool)
    for dec in range(2 ** A.shape[0]):
        b = utils.format.dec2bin(dec, A.shape[0])
        v = utils.format.bilinear(b, A)
        if v >= objective:
            objective = v
            gamma = b
    return objective, gamma

def test(d=10, scip=True, bf=True):
    '''
        Test run of SCIP/ZIMPL and brute force on a random instance.
        @param d dimension
        @param scip do SCIP
        @param bf do brute force
    '''
    A = utils.format.v2lt(numpy.random.normal(size=d * (d + 1) / 2))
    print 'test on random instance d=%d' % d
    if scip:
        t = time.time()
        scip_r = solve_scip(A)[1]
        print 'scip %.3fs' % (time.time() - t)
    if bf:
        t = time.time()
        bf_r = solve_bf(A)[1]
        print 'bf   %.3fs' % (time.time() - t)
    if scip and bf: assert(scip_r == bf_r).all()

def main():
    test(30, bf=False)

if __name__ == "__main__":
    main()

'''
# read dimension
param d := read "matrix.dat" as "1n" use 1;

# construct indices
set I   := { 1 .. d };
set T := { <i,j> in I * I with i <= j };
set S := { <i,j> in I * I with i <  j };

# read symmetric matrix
param A[T] := read "matrix.dat" as "1n" skip 1;

# declare variables binary
var x[T] binary;

# maximize target
maximize qb: sum <i,j> in T : A[i,j] * x[i,j];

# product constraints
subto or1: forall <i,j> in S do
   x[i,j] <= x[i,i];
subto or2: forall <i,j> in S do
   x[i,j] <= x[j,j];
subto and1: forall <i,j> in S do
   x[i,j] >= x[i,i] + x[j,j] - 1;
'''
