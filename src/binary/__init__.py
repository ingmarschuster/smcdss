#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

CONST_PRECISION = 1e-5
CONST_ITERATIONS = 50
CONST_MIN_MARGINAL_PROB = 1e-12

import scipy.stats as stats
import scipy
import numpy
import utils
import time

from product_model import ProductBinary
from logistic_cond_model import LogisticBinary
from qu_exponential_model import QuExpBinary
from qu_linear_model import QuLinearBinary
from gaussian_cop_model import GaussianCopulaBinary
from posterior_bvs import PosteriorBinary