#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from binary_model import Binary
from product_model import ProductBinary
from normal_model import HiddenNormalBinary
from logistic_model import LogisticBinary
from qu_exp_model import QuExpBinary
from linear_model import LinearBinary
from hybrid_model import HybridBinary
from posterior_bvs import PosteriorBinary

CONST_PRECISION = 1e-5
CONST_ITERATIONS = 50