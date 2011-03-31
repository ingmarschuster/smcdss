#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#    $Author: Christian Schäfer
#    $Date$

__version__ = "$Revision$"

import sys
import os
import ubqo
import numpy
import binary

from bf import solve_bf
from ce import solve_ce
from sa import solve_sa
from scip import solve_scip 

v = dict(

#---------------------------------------------------------------------------
# System configurations
#---------------------------------------------------------------------------

# The root directory of the project
SYS_ROOT=os.path.split(os.path.split(sys.path[0])[0])[0],

#---------------------------------------------------------------------------
# Cross entropy method
#---------------------------------------------------------------------------

# The number of particles to use for the Cross entropy method.
CE_N_PARTICLES = 4000,

# The binary model to used in th CE algorithm.
CE_BINARY_MODEL = binary.logistic_cond_model.LogisticBinary,

# The elite fraction used to estimate the next parameter.
CE_ELITE = 0.2,

# The lag in the parameter update.
CE_LAG = 0.3,


#---------------------------------------------------------------------------
# Simulated annealing
#---------------------------------------------------------------------------

# The Markov kernel to be used in the algorithm. Possible kernels are
# SymmetricMetropolisHastings, AdaptiveMetropolisHastings and Gibbs
SA_KERNEL = 'mh',

# The maximal running time in minutes.
SA_MAX_TIME = 30.0,

# The maximal number of iterations to perform.
SA_MAX_ITER = numpy.inf,


#---------------------------------------------------------------------------
# Running
#---------------------------------------------------------------------------

# The number of CPUs to be used. None implies that the algortihms are run
# without job server. The string 'autodetect' implies that all available
# CPUs are used.
RUN_CPUS=None,

# The default algorithm to run. 
RUN_ALGO=solve_bf,

# The default path to test run directory.
RUN_PATH='data/testopt',

# Write result into an output file.
RUN_OUTPUT=True,

# Write extensive information to stdout.
RUN_VERBOSE=False,

# The number of runs to be performed.
RUN_N=200

)
