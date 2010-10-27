'''
@author cschafer
'''

import binary
from numpy import zeros, array, diag, log, random
from auxpy.data import *

sample=data()
sample.load('/home/cschafer/Bureau/testfile.dump')

x = binary.logistic_binary.fromData(sample)
print sample.mean
print sample.cor
print
print x.marginals()

# [ 0.638  0.488  0.388  0.614  0.474]