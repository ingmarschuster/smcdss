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
