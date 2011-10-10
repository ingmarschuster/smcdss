#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Package for change point detection.
"""

"""
@namespace cpd
$Author: christian.a.schafer@gmail.com $
$Rev: 144 $
$Date: 2011-05-12 19:12:23 +0200 (jeu., 12 mai 2011) $
@details
"""

import os
import utils.conf_reader
import cpd_gen

CONST_PRECISION = 1e-8
v = dict()

# Read the configuration.
def read_config(file='cpd'):
    v.update(utils.conf_reader.read_config(file, v=v))