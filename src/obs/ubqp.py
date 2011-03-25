#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2011-02-17 18:34:45 +0100 (jeu., 17 févr. 2011) $
    $Revision: 73 $
'''

import numpy

import utils.format

def load_ubqp():
    filename = 'bqp50.txt'
    f = open('../../data/ubqp/' + filename, 'r')
    n = int(f.readline())
    d = int(f.readline().strip().split(' ')[0])

    A = numpy.zeros((d, d))
    line = f.readline().strip().split(' ')
    while len(line) == 3:
        A[int(line[1]) - 1, int(line[0]) - 1] = float(line[2])
        line = f.readline().strip().split(' ')
    f.close()
    print A.shape
    
    return A

load_ubqp()

