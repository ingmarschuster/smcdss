#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2011-02-17 18:34:45 +0100 (jeu., 17 févr. 2011) $
    $Revision: 73 $
'''

import numpy as np

filename = 'bqp50.txt'
f = open('../../data/ubqp/' + filename, 'r')
n = int(f.readline())
d = int(f.readline().strip().split(' ')[0])

A = np.array()

f.close()

print n, d
