#!/usr/bin/python
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#exit(0)

import os
import numpy

if os.name == 'nt':
    if os.environ.has_key('CPATH'):
        os.environ['CPATH'] = os.environ['CPATH'] + numpy.get_include()
    else:
        os.environ['CPATH'] = numpy.get_include()
    #mingw_setup_args = { 'options': { 'build_ext': { 'compiler': 'mingw32' } } }


ext_modules = [Extension(pyx_name, [pyx_name + '.pyx'])
               for pyx_name in ['base', 'product', 'pos_product', 'logistic_cond']]

setup(
  name='binary',
  cmdclass={'build_ext': build_ext},
  ext_package='binary',
  ext_modules=ext_modules
)

