#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2011-02-16 11:28:38 +0100 (mer., 16 févr. 2011) $
    $Revision: 71 $
'''

import numpy

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
    return name + '[' + ' '.join([('%.3f' % x).rjust(8) for x in v]) + ' ]\n'

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
    return ''.join([str(i) for i in numpy.array(bin, dtype=int)])

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
    return numpy.array(bin)

def tau(i, j):
    '''
        Maps the indices of a symmetric matrix onto the indices of a vector.
        @param i matrix index
        @param j matrix index
        @return vector index
    '''
    return j * (j + 1) / 2 + i

def m2v(A):
    '''
        Turns a symmetric matrix into a vector.
        @param matrix
        @return vector
    '''
    d = A.shape[0]
    a = numpy.zeros(d * (d + 1) / 2)
    for i in range(d):
        for j in range(i + 1):
            a[tau(i, j)] = A[i, j]
    return a

def v2m(a):
    '''
        Turns a vector into a symmetric matrix.
        @param vector
        @return matrix
    '''
    d = a.shape[0]
    d = int((numpy.sqrt(1 + 8 * d) - 1) / 2)
    A = numpy.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            A[i, j] = a[tau(i, j)]
            A[j, i] = A[i, j]
    return A

def v2lt(a):
    '''
        Turns a vector into a lower triangular matrix.
        @param vector
        @return matrix
    '''
    d = a.shape[0]
    d = int((numpy.sqrt(1 + 8 * d) - 1) / 2)
    A = numpy.zeros((d, d))
    k = 0
    for j in range(d):
        for i in range(j, d):
            A[i, j] = a[k]
            k += 1
    return A

def bilinear(v, A):
    return numpy.dot(numpy.dot(v, A), v)

