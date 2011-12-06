#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Package for optimization on binary spaces.
"""

"""
@namespace obs
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (lun., 10 oct. 2011) $
@details
"""

import os
import utils.conf_reader

CONST_PRECISION = 1e-8
v = dict()

# Read the configuration.
def read_config(file='obs'):
    v.update(utils.conf_reader.read_config(file, v=v))