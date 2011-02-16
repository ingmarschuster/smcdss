#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import time
import pyximport
pyximport.install()
import cython_primes, py_primes, weave_primes

n = 1e3
for (f, name) in [(cython_primes.primes, 'cython'), (py_primes.primes, 'python')]: #(weave_primes.primes, 'weave')
    t = time.clock()
    f(n)
    print '%s:\t%.3f' % (name, time.clock() - t)