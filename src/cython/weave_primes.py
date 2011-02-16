#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.weave import *
import numpy

def primes(kmax=10):
    code = \
    """
        int n, k, i;
        k = 0;
        n = 2;
        while (k < kmax) {
            i = 0;
            while (i < k && n % p(i) != 0) {
                i = i + 1;
            }
            if (i == k) {
                p(k) = n;
                k = k + 1;
            }
            n = n + 1;
        }
    """
    p = numpy.zeros(kmax, dtype=int)
    inline(code, ['p','kmax'], type_converters=converters.blitz, compiler='gcc')
    return list(p)
