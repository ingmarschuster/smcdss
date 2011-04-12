#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Package for optimization on binary spaces.
"""

"""
@namespace obs
$Author$
$Rev$
$Date$
@details
"""

import sys, os, time
import numpy

import ubqo
import binary
import utils
import ConfigParser

from numpy import inf
from bf import solve_bf, bf
from ce import solve_ce, ce
from sa import solve_sa, sa
from smc import solve_smc, smc
from scip import solve_scip, scip
from binary import LogisticBinary as logistic, ProductBinary as product, GaussianCopulaBinary as gaussian

v = {'SYS_ROOT':os.path.abspath(os.path.join(*([os.getcwd()] + ['..']*1)))}

def read_config(file=os.path.join(v['SYS_ROOT'], 'src', 'obs', 'default')):
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

    if not isinstance(v['RUN_PROBLEM'], list): v['RUN_PROBLEM'] = [v['RUN_PROBLEM']]