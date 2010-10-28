'''
@author cschafer
'''

import binary
from numpy import zeros, array, diag, log, random
from auxpy.data import *
from auxpy.plotting import *

x=binary.posteriorBinary('../data/datasets/test_dat.csv','hb')
plot4(x)