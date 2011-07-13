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

import os, sys
import ConfigParser

from numpy import inf
from smc import smc
from mcmc import mcmc, AdaptiveMetropolisHastings as adaptive, SymmetricMetropolisHastings as symmetric, SwapMetropolisHastings as swap, Gibbs as gibbs
from binary import LogisticBinary as logistic, ProductBinary as product

CONST_PRECISION = 1e-8

v = {'SYS_ROOT':os.path.abspath(os.path.join(*([os.getcwd()] + ['..']*1)))}

def read_config(file=os.path.join(v['SYS_ROOT'], 'src', 'ibs', 'default')):

    config = ConfigParser.SafeConfigParser()
    config.read(file + '.ini')

    for s_key in config._sections.keys():
        for e_key in config._sections[s_key].keys():
            key = e_key.upper()
            v.update({key:config._sections[s_key][e_key]})
            if v[key] == '': v[key] = None
            try:
                v[key] = eval(v[key])
            except:
                pass

    if not os.path.isabs(v['RUN_PATH']):
        v['RUN_PATH'] = os.path.join(v['SYS_ROOT'], os.path.normpath(v['RUN_PATH']))