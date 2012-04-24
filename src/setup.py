#!/usr/bin/python
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
import numpy

# set C path for windows
if os.name == 'nt':
    if os.environ.has_key('CPATH'): os.environ['CPATH'] = os.environ['CPATH'] + numpy.get_include()
    else: os.environ['CPATH'] = numpy.get_include()

    '''
    mingw_setup_args = { 'options': { 'build_ext': { 'compiler': 'mingw32' } } }
    pyximport.install(setup_args=mingw_setup_args, build_dir=os.path.curdir)
    '''

# extensions
ext_modules = [Extension('binary.%s' % pyx_name, ['binary/%s.pyx' % pyx_name])
               for pyx_name in ['base', 'product', 'product_positive', 'product_constrained',
                                'selector_glm', 'selector_glm_ml', 'selector_glm_bayes',
                                'product_limited', 'conditionals', 'conditionals_logistic', 'quadratic_linear']]

ext_modules += [Extension('algo.resample', ['algo/resample.pyx'])]

# build directory
if os.name == 'posix':
    build_temp = os.path.join('..', 'build', '_'.join(list(os.uname())[0::2]))
else:
    build_temp = os.path.join('..', 'build', 'win32')

# run setup
setup(
  name='binary',
  author='Christian Schäfer',
  options={'build_ext': {'inplace': True, 'build_temp':build_temp}},
  cmdclass={'build_ext': build_ext},
  ext_modules=ext_modules
)
