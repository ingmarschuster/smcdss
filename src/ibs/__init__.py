from binary import ProductBinary, LogisticBinary
from numpy import inf

#---------------------------------------------------------------------------
# Sequential Monte Carlo
#---------------------------------------------------------------------------

# The number of particles to use for Sequential Monte Carlo. 
SMC_N_PARTICLES = 20000

# The minimum distance of the marginal probability from the
# boudaries of the unit interval. For details see
# arxiv.org/pdf/1101.6037, page 8, section "Sparse version of the model"
SMC_EPS = 0.02

# The minimum correlation required to include the component in a logistic
# regression. For details see arxiv.org/pdf/1101.6037, page 8, paragraph
# "Sparse version of the model"
SMC_DELTA = 0.075

# The efficient sample size targeted when computing the step length.
# For details see arxiv.org/pdf/1101.6037, page 5, section
# "Finding the step length"
SMC_ETA = 0.9

# The binary model to used in th SMC algorithm. This should either be
# LogisticBinary or ProductBinary.
SMC_BINARY_MODEL = LogisticBinary


#---------------------------------------------------------------------------
# Markov chain Monte Carlo
#---------------------------------------------------------------------------

# The Markov kernel to be used in the algorithm. Possible kernels are
# SymmetricMetropolisHastings, AdaptiveMetropolisHastings and Gibbs
MCMC_KERNEL = SymmetricMetropolisHastings

# The maximum number of iterations to perform.
MCMC_MAX_EVALS = 2e6

# The average number of bit to be flipped at a time.
MCMC_Q = 1


#---------------------------------------------------------------------------
# Data processing
#---------------------------------------------------------------------------

# The data set to perform the variable selection on.
DATA_SET = 'boston',

# The column of the first covariate.
DATA_FIRST_COVARIATE = 2

# The number of covariates to include in the variable selection problem.
DATA_N_COVARIATES = inf

# The default path to data set directory.
DATA_PATH = 'data/datasets'

# The type of model selection. Possible choices are 'hb' for a conjugate
# Hierarchical Bayesian setup or 'bic' for the Bayesian Information Criterion
POSTERIOR_TYPE = 'hb'


#---------------------------------------------------------------------------
# Running
#---------------------------------------------------------------------------

# The number of CPUs to be used. None implies that the algortihms are run
# without job server. The string 'autodetect' implies that all available
# CPUs are used.
RUN_N_CPUS = None

# The default algorithm to run. 
RUN_ALGO = smc

# The default path to test run directory.
RUN_PATH = 'data/testruns'

# Write result into an output file.
RUN_OUTPUT = True

# Write extensive information to stdout.
RUN_VERBOSE = False

# The number of runs to be performed.
RUN_N = 200


#---------------------------------------------------------------------------
# Evaluation
#---------------------------------------------------------------------------

# Layout on A4 landscape.
EVAL_A4 = False

# The height of the R graph.
EVAL_HEIGTH = 4

# The width of the R graph.
EVAL_WIDTH = 12

# Show names of covariates. Otherwise the columns are numbered.
EVAL_NAMES = False

# Print title line on top of the R graph.
EVAL_TITLE = False

# The percentage of data to be contained in the box.
EVAL_BOXPLOT = 0.8

# The color used for the boxplot.
EVAL_COLOR = ['azure1', 'black', 'white', 'white', 'black']

# The outer margins (bottom, left, top, right) 
EVAL_OUTER_MARGIN = [0, 0, 0, 0],

# The inner margins (bottom, left, top, right) 
EVAL_INNER_MARGIN = [2, 2, 0.5, 0]

# The font family.      
EVAL_FONT_FAMILY = 'serif'

# The font family size.
EVAL_FONT_CEX = 1

# The number of lines below the main title.
EVAL_TITLE_LINE = 1