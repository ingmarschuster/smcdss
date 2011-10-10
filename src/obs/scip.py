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
import obs
import binary
import numpy
import os
import sys
import time
import ubqo
import fileinput

class scip(ubqo.ubqo):
    name = 'SCIP'
    header = []
    def run(self):
               
        primal_bound, dual_bound, solution = solve_scip(f=binary.QuExpBinary(self.A), testsuite=self.testsuite, problem=self.problem, tl=self.v['SCIP_TIME_LIMIT'])
        
        if primal_bound > self.primal_bound:self.primal_bound = primal_bound
        if dual_bound < self.dual_bound:self.dual_bound = dual_bound
        
        # save best objective and dual bound obtained
        for i, line in enumerate(fileinput.FileInput(os.path.join(obs.v['DATA_PATH'], self.testsuite, self.testsuite + '_%02d.dat' % self.problem), inplace=1)):
            if i == 0: line = '%.f\n' % primal_bound
            if i == 1: line = '%.10f\n' % dual_bound
            print line,
        
        return solution

def solve_scip(f, testsuite, problem, tl=None):
    """
        Solve UBQO using SCIP and ZIMPL from the ZIB Optimization Suite http://zibopt.zib.de.
        @param f quadratic exponential model.
        @return best_obj maximum
        @return best_soln maximizer
    """

    t = time.time()
    print 'Running scip 2.01 using zimpl 3.1.0.\n'

    scip_path = os.path.join(obs.v['SYS_ROOT'], 'bin', 'scip')
    
    # set time limit
    setting_path = os.path.join(scip_path, 'parameters_uqbo.set')
    if tl is None: tl = 1e20
    file = open(setting_path, 'w')
    file.write("limits/time=%f\n" % (tl * 60))
    file.write("limits/memory=2000")
    file.close()

    uqbo_path = os.path.join(scip_path, 'temp', 'uqbo_%s_%02d_%06d' % (testsuite, problem, numpy.random.randint(low=0, high=1e7 - 1))) + '%s'

    # create ZPL file
    file = open(uqbo_path % '.zpl', 'w')
    file.write(ZIMPL_QUADRATIC % {'datfile':os.path.join(obs.v['DATA_PATH'], testsuite, testsuite + '_%02d.dat' % problem)})
    file.close()
        
    # invoke scip
    if os.path.exists(uqbo_path % '.log'): os.remove(uqbo_path % '.log')
    if os.name == 'posix':
        subprocess.Popen([os.path.expanduser('~/ziboptsuite-2.0.1/scip-2.0.1/bin/scip'), '-l', (uqbo_path % '.log'), '-s', setting_path, '-f', (uqbo_path % '.zpl'), '-q']).wait()
    else:
        # invoke zimpl separately
        bin_path = os.path.join(obs.v['SYS_ROOT'], 'bin', 'scip')
        cwd_path = os.getcwd()
        os.chdir(os.path.join(bin_path, 'temp'))
        subprocess.Popen([os.path.join(bin_path, 'zimpl.exe'), '-t', 'pip', '-v', '0', os.path.basename(uqbo_path % '.zpl')]).wait()
        os.chdir(cwd_path)
        p = subprocess.Popen([os.path.join(bin_path, 'scip.exe'), '-l', (uqbo_path % '.log'), '-s', setting_path, '-f', (uqbo_path % '.pip'), '-q'])

    header = " time | node  | left  |LP iter|LP it/n| mem |mdpt |frac |vars |cons |cols |rows |cuts |confs|strbr|  dualbound   | primalbound  |  gap   "
    print header[:7] + header[-38:]
    while True:    
        time.sleep(1.0)
        file = open((uqbo_path % '.log'), 'r')
        result = file.read().split('\n')[-2]
        file.close()
        sys.stdout.write('\r' + result[:7] + result[-38:])
        if p.poll() is not None: break
    sys.stdout.write('\r\n\n')
    
    # read log
    file = open((uqbo_path % '.log'), 'r')
    result = file.read()
    file.close()
    
    result = result.split('primal solution:\n================\n\n', 2)[1]
    result = result.split('\n\nStatistics\n==========\n', 2)

    # retrieve best solution
    sol = result[0].split('\n')
    best_soln = numpy.zeros(f.d, dtype=bool)
    for x in sol[1:]:
        x = x.split()[0].split('#')
        if x[0] == 'z': break
        best_soln[int(x[1]) - 1] = True
        
    # retrieve bounds
    bounds = result[1].split('Solution           :\n', 2)[1].split('\n')
    primal_bound, dual_bound = float(bounds[2][21:45]), float(bounds[3][21:45])
    
    print 'objective : %.1f ' % primal_bound
    print 'dual bound: %.1f ' % dual_bound
    print 'opt gap   : %.3f %%' % [100.0, 100 * (dual_bound - primal_bound) / dual_bound][dual_bound <> 0]

    # clean up temp folder
    for suffix in ['.zpl', '.pip', '.tbl', '.log']:
        if os.path.exists(uqbo_path % suffix): os.remove(uqbo_path % suffix)

    return primal_bound, dual_bound, {'obj' : primal_bound, 'soln' : best_soln, 'time' : time.time() - t}

def main():
    pass

if __name__ == "__main__":
    main()

ZIMPL_QUADRATIC = """
# Read dimension.
param d := read "%(datfile)s" as "1n" use 1 skip 2;

# Construct indices.
set I := { 1 .. d };
set T := { <i,j> in I * I with i <= j };
set S := { <i,j> in I * I with i <  j };

# Read symmetric matrix.
param A[T] := read "%(datfile)s" as "1n" skip 3;

# Declare variables binary.
var x[I] binary;
var z integer;

# Maximize target.
maximize qb: z;

# Product constraints
subto name : z == sum <i,j> in T : A[i,j] * x[i] * x[j];
"""

ZIMPL_LINEARIZED = """
# Read dimension.
param d := read "%(datfile)s" as "1n" use 1 skip 2;

# Construct indices.
set I := { 1 .. d };
set T := { <i,j> in I * I with i <= j };
set S := { <i,j> in I * I with i <  j };

# Read symmetric matrix.
param A[T] := read "%(datfile)s" as "1n" skip 3;

# Declare variables binary.
var x[T] binary;

# Maximize target.
maximize qb: sum <i,j> in T : A[i,j] * x[i,j];

# Product constraints.
subto or1: forall <i,j> in S do
   x[i,j] <= x[i,i];
subto or2: forall <i,j> in S do
   x[i,j] <= x[j,j];
subto and1: forall <i,j> in S do
   x[i,j] >= x[i,i] + x[j,j] - 1;
"""

"""
Example on how to extract relaxation from optimization task:

 read /mnt/fat32/Crest/Python/smcdss/src/obs/scip/uqbo_r250u_neu.zpl
 presolve
 set separating emphasis off
 set limit node 1
 optimize
 write lp lp.lp
 read lp.lp
 read variablen.fix
 optimize
 write solution ergebnis.sol
"""

