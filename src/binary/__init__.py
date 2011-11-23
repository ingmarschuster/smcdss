#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Package containing various kinds of models for multivariate binary data.
"""

"""
@namespace binary
$Author$
$Rev$
$Date$
@details The models are all derived from the base class Binary.
"""

"""
@mainpage Sequential Monte Carlo algorithms on binary spaces

@section binary_sec Binary models

The package @link binary binary @endlink for various binary models.

@section int_sec Integration

The package @link ibs @endlink for Monte Carlo integration.
 
@section opt_sec Optimization

The package @link obs @endlink for Monte Carlo optimization.
"""

CONST_PRECISION = 1e-5
CONST_ITERATIONS = 50
CONST_MIN_MARGINAL_PROB = 1e-10

import scipy.stats as stats
import scipy
import numpy
import utils
import time

from product_model import ProductBinary
from logistic_cond_model import LogisticBinary
from uniform_model import UniformBinary
from posterior import Posterior

try:
    from qu_exponential_model import QuExpBinary
    from qu_linear_model import QuLinearBinary
    from gaussian_cop_model import GaussianCopulaBinary
except:
    pass