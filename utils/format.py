#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Formatting console output.
"""

"""
@namespace utils.format
$Author: christian.a.schafer@gmail.com $
$Rev: 144 $
$Date: 2011-05-12 19:12:23 +0200 (jeu., 12 mai 2011) $
@details
"""

import numpy, datetime
import sys

def format(X, name=''):
    """
        Formats a vector or matrix for output on stdout
        @param X vector or matrix
        @param name name 
    """
    if len(X.shape) == 1: return format_vector(X, name)
    if len(X.shape) == 2: return format_matrix(X, name)

def format_vector(v, name=''):
    """
        Formats a vector for output on stdout
        @param v vector 
        @param name name 
    """
    if not name == '': name = name + ' =\n'
    return name + '[' + ' '.join([('%.3f' % x).rjust(8) for x in v]) + ' ]\n'

def format_matrix(M, name=''):
    """
        Formats a matrix for output on stdout
        @param M matrix
        @param name name 
    """
    if not name == '': name = name + ' =\n'
    return name + ''.join([format_vector(x) for x in M])

def bin2str(bin):
    """
        Converts a boolean array to a string representation.
        @param bin boolean array 
    """
    return ''.join([str(i) for i in numpy.array(bin, dtype=int)])

def bin2dec(bin):
    """
        Converts a boolean array into an integer.
        @param bin boolean array 
    """
    return long(bin2str(bin), 2)

def dec2bin(n, d=0):
    """
        Converts an integer into a boolean array containing its binary representation.
        @param n integer
        @param d dimension of boolean vector
    """
    bin = []
    while n > 0:
        if n % 2: bin.append(True)
        else: bin.append(False)
        n = n >> 1
    while len(bin) < d: bin.append(False)
    bin.reverse()
    return numpy.array(bin)

def bilinear(v, A):
    return numpy.dot(numpy.dot(v, A), v)

def isnumeric(s):
    return s.startswith('-') and s[1:].isdigit() or s.isdigit()

def time(seconds):
    return str(datetime.timedelta(seconds=seconds)).split('.')[0]

def progress(ratio, text=None, ticks=50, last_ratio=0):
    if abs(ratio - last_ratio) * 101.0 < 0.1: return last_ratio
    progress = int(ticks * ratio)
    s = '%.1f%%' % (100.0 * ratio)
    length = len(s)
    if progress > ticks / 2 - length:
        sys.stdout.write('\r[' + (ticks / 2 - length) * '-' + s
                         + (progress - ticks / 2) * '-' + min(ticks - progress, ticks / 2) * ' ' + ']')
    else:
        sys.stdout.write('\r[' + progress * '-' + (ticks / 2 - length - progress) * ' ' + s
                         + (ticks / 2) * ' ' + ']')
    if not text is None: sys.stdout.write(str(text))
    sys.stdout.flush()
    return ratio

