from ceopt import *
from numpy import inf
from binary import ProductBinary, LogisticBinary

#---------------------------------------------------------------------------
# Cross entropy method
#---------------------------------------------------------------------------

# The number of particles to use for the Cross entropy method.
CE_N = 5000

# The binary model to used in th SMC algorithm. This should either be
# LogisticBinary or ProductBinary.
CE_BINARY_MODEL = LogisticBinary

# The elite fraction used to estimate the mean.
CE_ELITE = 0.5

# The lag in the mean update.
CE_LAG = 0.2

# Epsilon.
CE_EPS = 0.075

# Delta.
CE_DELTA = 0.1


#---------------------------------------------------------------------------
# Simulated annealing
#---------------------------------------------------------------------------

# The Markov kernel to be used in the algorithm. Possible kernels are
# SymmetricMetropolisHastings, AdaptiveMetropolisHastings and Gibbs
SA_KERNEL = 'mh'

# The maximal running time in minutes.
SA_MAX_TIME = 30.0

# The maximal number of iterations to perform.
SA_MAX_ITER = inf