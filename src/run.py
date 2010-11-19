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

#x=binary.PosteriorBinary('../data/datasets/test_dat.csv','bic')
#plot4(x)

x = binary.LogLinearBinary.random(12,0.01)
print x.marginals()

y = binary.LogisticRegrBinary.from_loglinear_model(x)
print y.marginals()

