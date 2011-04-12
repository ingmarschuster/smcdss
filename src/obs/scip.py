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
        return solve_scip(f=binary.QuExpBinary(self.A))

def solve_scip_linux(f):
    """
        Solve UBQO using SCIP and ZIMPL from the ZIB Optimization Suite http://zibopt.zib.de.
        @param f quadratic exponential model.
        @return best_obj maximum
        @return best_soln maximizer
    """

    print 'running scip 2.01 using zimpl 3.1.0'
    t = time.time()
    # write matrix
    file = open('obs/scip/matrix.dat', 'w')
    file.write('%i\n' % f.d)
    for j in xrange(f.d):
        for i in xrange(j, f.d):
            if i == j: a = f.A[i, j]
            else: a = 2 * f.A[i, j]
            file.write('%.16f\n' % a)
    file.close()

    # invoke scip
    if os.path.exists('obs/scip/scip.log'): os.remove('obs/scip/scip.log')
    subprocess.Popen(['/home/cschafer/ziboptsuite-2.0.1/scip-2.0.1/bin/scip', '-l', 'obs/scip/scip.log', '-f', 'obs/scip/uqbo.zpl', '-q']).wait()

    # read log
    file = open('obs/scip/scip.log', 'r')
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

    print 'objective: %.1f ' % best_obj
    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def solve_scip_win(f):
    """
        Solve UBQO using SCIP and ZIMPL from the ZIB Optimization Suite http://zibopt.zib.de.
        @param f quadratic exponential model.
        @return best_obj maximum
        @return best_soln maximizer
    """

    print 'running scip 2.01 using zimpl 3.1.0'
    t = time.time()
    # write matrix
    file = open('obs\\scip\\matrix.dat', 'w')
    file.write('%i\n' % f.d)
    for j in xrange(f.d):
        for i in xrange(j, f.d):
            if i == j: a = f.A[i, j]
            else: a = 2 * f.A[i, j]
            file.write('%.16f\n' % a)
    file.close()

    # invoke scip
    if os.path.exists('obs\\scip\\scip_win.log'): os.remove('obs\\scip\\scip_win.log')
    bin_path = os.path.join('W:\\', 'Documents', 'Python', 'smcdss', 'bin')
    cwd_path = os.getcwd()
    os.chdir(os.path.join('obs', 'scip'))
    subprocess.Popen([os.path.join(bin_path, 'zimpl.exe'), '-O', '-v 0' , 'uqbo.zpl']).wait()
    os.chdir(cwd_path)
    subprocess.Popen([os.path.join(bin_path, 'scip.exe'), '-l', 'obs\\scip\\scip_win.log', '-f', 'obs\\scip\\uqbo.lp', '-q']).wait()

    # read log
    file = open('obs\\scip\\scip_win.log', 'r')
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

    print 'objective: %.1f ' % best_obj
    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

if os.name == 'posix': solve_scip = solve_scip_linux
else: solve_scip = solve_scip_win

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
