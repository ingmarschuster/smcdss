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

g=LogisticRegrBinary.random(60,scale=3)

d=data()
d.sample(g,5000)

l=LogisticRegrBinary.from_data(d,verbose=True)
print l.getModelSize()
