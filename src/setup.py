#!/usr/bin/python
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
import shutil
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
               for pyx_name in ['base', 'product', 'pos_product', 'logistic_cond']]

# build directory
build_temp = os.path.join('..', 'build', '_'.join(list(os.uname())[0::2]))

# run setup
setup(
  name='binary',
  author='Christian Schäfer',
  options={'build_ext': {'inplace': True, 'build_temp':build_temp}},
  cmdclass={'build_ext': build_ext},
  ext_modules=ext_modules
)

# move C files
for filename in os.listdir('binary'):
    if filename[-2:] == '.c':
        shutil.move(os.path.join('binary', filename),
                    os.path.join(build_temp, 'binary', filename))
