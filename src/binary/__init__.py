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