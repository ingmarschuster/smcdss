#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

import binary
from numpy import zeros, array, diag, log, random
from auxpy.data import *
from auxpy.plotting import *
from datetime import time
from algos.ceopt import ceopt
from algos.smc import *

def testceopt():
    target = PosteriorBinary(dataFile='/home/cschafer/Documents/smcdss/data/datasets/test_dat.csv')
    max, time = ceopt(target, verbose=True)
    print '\n[' + ', '.join([str(i) for i in where(max['state'])[0]]) + ']',
    print '\ntime %.2f' % time

def testsmc():
    target = PosteriorBinary(dataFile='/home/cschafer/Documents/smcdss/data/datasets/test_dat.csv')
    mean, time = smc(target, verbose=False)
    print format(mean)
    print '\ntime %.2f' % time

testsmc()
