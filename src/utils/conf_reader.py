#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Parameter reader.
"""

"""
@namespace utils.cython
$Author: christian.a.schafer@gmail.com $
$Rev: 122 $
$Date: 2011-04-12 19:22:11 +0200 (mar., 12 avr. 2011) $
@details
"""

import os
import ConfigParser

from numpy import inf

# binary parametric families
from binary import LogisticBinary as logistic, ProductBinary as product, GaussianCopulaBinary as gaussian

# integration algorithms
from ibs.smc import smc, univariate
from ibs.mcmc import mcmc, AdaptiveMetropolisHastings as adaptive, SymmetricMetropolisHastings as symmetric, SwapMetropolisHastings as swap, Gibbs as gibbs

# optimization algorithms
try:
    from obs.bf import solve_bf, bf
    from obs.ce import solve_ce, ce
    from obs.sa import solve_sa, sa
    from obs.smca import solve_smca, smca
    from obs.scip import solve_scip, scip
except:
    pass

def read_config(file, v):
    """
        Reads a configuration file and overwrite the default values.
        \param file valid INI-file
    """

    # locate project root folder
    path = os.getcwd()
    while not os.path.basename(path) == 'smcdss':
        path = os.path.abspath(os.path.join(*([path] + ['..'] * 1)))
    v.update({'SYS_ROOT':path})

    # load default config files
    if file in ['ibs', 'obs', 'cpd']:
        file = os.path.join(path, 'src', file + '_default')
    if not file[-4:] == '.ini':file += '.ini'

    config = ConfigParser.SafeConfigParser()
    config.read(file)

    for s_key in config._sections.keys():
        for e_key in config._sections[s_key].keys():
            key = e_key.upper()
            v.update({key:config._sections[s_key][e_key]})
            if v[key] == '': v[key] = None
            try:
                v[key] = eval(v[key])
            except:
                pass

    if os.name == 'posix': OS = 'POSIX_'
    else: OS = 'WIN32_'

    v.update({'RUN_PATH':v[OS + 'RUN_PATH'],
              'DATA_PATH':v[OS + 'DATA_PATH'],
              'SYS_R':v[OS + 'R'],
              'SYS_VIEWER':v[OS + 'VIEWER']
              })

    for PATH in ['RUN_PATH', 'DATA_PATH', 'SYS_R', 'SYS_VIEWER']:
        if not os.path.isabs(v[PATH]):
            v[PATH] = os.path.join(v['SYS_ROOT'], os.path.normpath(v[PATH]))

    return v
