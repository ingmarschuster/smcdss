#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2011-02-17 11:53:17 +0100 (jeu., 17 févr. 2011) $
    $Revision: 72 $
'''

import time

from binary import *
import utils
import pp

d = 500
n = 50000
b = ProductBinary.random(d)
s = utils.data.data()

t = time.clock()
b.rvs(size=n, job_server=pp.Server(ncpus=2, ppservers=()))
print 'job server %.3f sec' % (time.clock() - t)

t = time.clock()
b.rvs(size=n, job_server=None)
print 'no job server %.3f sec' % (time.clock() - t)


