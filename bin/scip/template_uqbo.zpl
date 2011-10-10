# Read dimension.
param d := read "matrix%(ts)s.dat" as "1n" use 1;

# Construct indices.
set I := { 1 .. d };
set T := { <i,j> in I * I with i <= j };
set S := { <i,j> in I * I with i <  j };

# Read symmetric matrix.
param A[T] := read "matrix%(ts)s.dat" as "1n" skip 1;


#
# Advanced formulation.
#

# Declare variables binary.
var x[I] binary;
var z integer;

# Maximize target.
maximize qb: z;

# Product constraints
subto name : z == sum <i,j> in T : A[i,j] * x[i] * x[j];


#
# Straightforward formulation.
#
# Declare variables binary.
#var x[T] binary;

# Maximize target.
#maximize qb: sum <i,j> in T : A[i,j] * x[i,j];

# Product constraints.
#subto or1: forall <i,j> in S do
#   x[i,j] <= x[i,i];
#subto or2: forall <i,j> in S do
#   x[i,j] <= x[j,j];
#subto and1: forall <i,j> in S do
#   x[i,j] >= x[i,i] + x[j,j] - 1;


#
# Extract relaxation from optimization task
#
# read /mnt/fat32/Crest/Python/smcdss/src/obs/scip/uqbo_r250u_neu.zpl
# presolve
# set separating emphasis off
# set limit node 1
# optimize
# write lp lp.lp
# read lp.lp
# read variablen.fix
# optimize
# write solution ergebnis.sol
