#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
SCIP.
"""

"""
@namespace obs.scip
$Author$
$Rev$
$Date$
@details
"""

import subprocess
from obs import *

class scip(ubqo.ubqo):
    name = 'SCIP'
    header = []
    def run(self):
        self.best_obj, solution = solve_scip(f=binary.QuExpBinary(self.A), ts=self.v['RUN_TESTSUITE'])
        return solution

def solve_scip(f, ts=''):
    """
        Solve UBQO using SCIP and ZIMPL from the ZIB Optimization Suite http://zibopt.zib.de.
        @param f quadratic exponential model.
        @return best_obj maximum
        @return best_soln maximizer
    """

    t = time.time()
    if not ts == '': ts = '_' + ts
    print 'running scip 2.01 using zimpl 3.1.0'

    # write matrix
    file = open(os.path.normpath('obs/scip/matrix%s.dat' % ts), 'w')
    file.write('%i\n' % f.d)
    for j in xrange(f.d):
        for i in xrange(j, f.d):
            if i == j: a = f.A[i, j]
            else: a = 2 * f.A[i, j]
            file.write('%.16f\n' % a)
    file.close()

    # write zimpl file
    file = open('obs/scip/uqbo.zpl', 'r')
    zpl = file.read()
    file.close()
    file = open('obs/scip/uqbo%s.zpl' % ts, 'w')
    file.write(zpl % {'ts':ts})
    file.close()

    # invoke scip
    if os.path.exists(os.path.normpath('obs/scip/scip%s.log' % ts)):
        os.remove(os.path.normpath('obs/scip/scip%s.log' % ts))
    if os.name == 'posix':
        subprocess.Popen(['/home/cschafer/ziboptsuite-2.0.1/scip-2.0.1/bin/scip', '-l', 'obs/scip/scip%s.log' % ts, '-f', 'obs/scip/uqbo%s.zpl' % ts, '-q']).wait()
    else:
        bin_path = os.path.join('W:\\', 'Documents', 'Python', 'smcdss', 'bin')
        cwd_path = os.getcwd()
        os.chdir(os.path.join('obs', 'scip'))
        subprocess.Popen([os.path.join(bin_path, 'zimpl.exe'), '-O', '-v 0' , 'uqbo%s.zpl' % ts]).wait()
        os.chdir(cwd_path)
        subprocess.Popen([os.path.join(bin_path, 'scip.exe'), '-l', 'obs\\scip\\scip%s.log' % ts, '-f', 'obs\\scip\\uqbo%s.lp' % ts, '-q']).wait()

    # read log
    file = open(os.path.normpath('obs/scip/scip%s.log' % ts), 'r')
    s = file.read()
    file.close
    s = s.split('primal solution:\n================\n\n', 2)[1]
    s = s.split('\n\nStatistics\n==========\n', 2)[0]
    s = s.split('\n')

    # retrieve best objective
    best_obj = float(s[0][16:].strip())

    # retrieve best solution
    best_soln = numpy.zeros(f.d, dtype=bool)
    for x in s[1:]:
        x = x.split()[0].split('#')[1:]
        if x[0] == x[1]: best_soln[int(x[0]) - 1] = True

    for file in ['scip%s.log', 'matrix%s.dat', 'uqbo%s.zpl', 'uqbo%s.lp', 'uqbo%s.tbl']:
        if os.name == 'posix':
            path = os.path.normpath('obs/scip/%s' % file % ts)
        else:
            path = os.path.normpath('W:\\Documents/Python/smcdss/src/obs/scip/%s' % file % ts)
        if os.path.exists(path): os.remove(path)

    print 'objective: %.1f ' % best_obj
    return best_obj, {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def main():
    pass

if __name__ == "__main__":
    main()

"""
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
"""
