import ubqo
import numpy
import binary
from brute_force import solve_bf
from scip import solve_scip 

#---------------------------------------------------------------------------
# Cross entropy method
#---------------------------------------------------------------------------

# The number of particles to use for the Cross entropy method.
CE_N_PARTICLES = 4000

# The binary model to used in th CE algorithm.
CE_BINARY_MODEL = binary.logistic_cond_model.LogisticBinary
#CE_BINARY_MODEL = binary.product_model.ProductBinary

# The elite fraction used to estimate the mean.
CE_ELITE = 0.2

# The lag in the mean update.
CE_LAG = 0.3


#---------------------------------------------------------------------------
# Simulated annealing
#---------------------------------------------------------------------------

# The Markov kernel to be used in the algorithm. Possible kernels are
# SymmetricMetropolisHastings, AdaptiveMetropolisHastings and Gibbs
SA_KERNEL = 'mh'

# The maximal running time in minutes.
SA_MAX_TIME = 30.0

# The maximal number of iterations to perform.
SA_MAX_ITER = numpy.inf