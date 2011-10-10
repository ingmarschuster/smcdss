#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Package for integration on binary spaces.
"""

"""
@namespace ibs
$Author$
$Rev$
$Date$
@details The packages includes algorithms based on classic Markov chain Monte
Carlo and novel Sequential Monte Carlo algorithms using particle methods.
"""

import os
import utils.conf_reader

CONST_PRECISION = 1e-8
v = dict()

# Read the configuration.
def read_config(file='ibs'):
    v.update(utils.conf_reader.read_config(file, v=v))
