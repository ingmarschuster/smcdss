#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2011-02-16 11:28:38 +0100 (mer., 16 févr. 2011) $
    $Revision: 71 $
'''

from numpy import *

def format(X, name=''):
    '''
        Formats a vector or matrix for output on stdout
        @param X vector or matrix
        @param name name 
    '''
    if len(X.shape) == 1: return format_vector(X, name)
    if len(X.shape) == 2: return format_matrix(X, name)

def format_vector(v, name=''):
    '''
        Formats a vector for output on stdout
        @param v vector 
        @param name name 
    '''
    if not name == '': name = name + ' =\n'
    return name + '[' + ' '.join([('%.4f' % x).rjust(7) for x in v]) + ' ]\n'

def format_matrix(M, name=''):
    '''
        Formats a matrix for output on stdout
        @param M matrix
        @param name name 
    '''
    if not name == '': name = name + ' =\n'
    return name + ''.join([format_vector(x) for x in M])

def bin2str(bin):
    '''
        Converts a boolean array to a string representation.
        @param bin boolean array 
    '''
    return ''.join([str(i) for i in array(bin, dtype=int)])

def bin2dec(bin):
    '''
        Converts a boolean array into an integer.
        @param bin boolean array 
    '''
    return long(bin2str(bin), 2)

def dec2bin(n, d=0):
    '''
        Converts an integer into a boolean array containing its binary representation.
        @param n integer
        @param d dimension of boolean vector
    '''
    bin = []
    while n > 0:
        if n % 2: bin.append(True)
        else: bin.append(False)
        n = n >> 1
    while len(bin) < d: bin.append(False)
    bin.reverse()
    return array(bin)