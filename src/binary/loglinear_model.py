#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-10-29 13:41:14 +0200 (ven., 29 oct. 2010) $
    $Revision: 29 $
'''

from numpy import *
from auxpy.data import *

tau = lambda i, j: j * (j + 1) / 2 + i

def m2v(A):
    d = A.shape[0]
    a = zeros(d * (d + 1) / 2)
    for i in range(d):
        for j in range(i, d):
            a[tau(i, j)] = A[i, j]
    return a

def v2m(a):
    d = a.shape[0]
    d = int((sqrt(1 + 8 * d) - 1) / 2)
    A = zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            A[i, j] = a[tau(i, j)]
            A[j, i] = A[i, j]
    return A

def calc_A(S):
    d = S.shape[0]
    dd = d * (d + 1) / 2 + 1;
    x_tau = ones((dd, dd))
    for i in range(d):
        for j in range(i, d):
            for k in range(d):
                for l in range(k, d):
                    x_tau[tau(i, j), tau(k, l)] = 2 * (
    (1 + (k == i | k == j)) * (1 + (l == i | l == j)) * (l != k) + (1 + (k == i | k == j)) * (l == k))

    s = m2v(S);
    s = 2 * array(list(s) + [1.0])

    x_tau[:dd - 1, :dd - 1] /= 4.0
    x_tau[dd - 1, dd - 1] = 2.0

    print format(x_tau)

    a = linalg.solve(x_tau, s)

    pi0 = a[dd - 1]
    a = a[:dd - 1]
    A = v2m(a)
    c = 2 ** (d - 1)

    print A


d = 2
mean = ones(d) * 0.01
R = eye(d)

#R = random.random((d, d)) 
#R = dot(R.T, R)
#r = diag(R)
#R /= sqrt(dot(r[:, newaxis], r[newaxis, :]))

var = mean * (1 - mean)
S = R * sqrt(dot(var[:, newaxis], var[newaxis, :])) + dot(mean[:, newaxis], mean[newaxis, :])

calc_A(S)
