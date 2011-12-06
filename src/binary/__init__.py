#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Parametric families for sampling random multivariate binary data. """

"""
\namespace binary
$Author$
$Rev$
$Date$
"""

"""
import os
import numpy
import pyximport

if os.name == 'nt':
    if os.environ.has_key('CPATH'):
        os.environ['CPATH'] = os.environ['CPATH'] + numpy.get_include()
    else:
        os.environ['CPATH'] = numpy.get_include()
    mingw_setup_args = { 'options': { 'build_ext': { 'compiler': 'mingw32' } } }
    pyximport.install(setup_args=mingw_setup_args, build_dir=os.path.curdir)
else:
    pyximport.install()

from base import BaseBinary
from product import ProductBinary
from logistic_cond import LogisticCondBinary
from uniform import UniformBinary
from posterior import Posterior
from qu_exponential import QuExpBinary
from qu_linear import QuLinearBinary
from gaussian_copula import GaussianCopulaBinary
"""
