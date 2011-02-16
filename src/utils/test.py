#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-12-10 19:39:02 +0100 (ven., 10 déc. 2010) $
    $Revision: 0 $
'''

import time, sys
import numpy as np

import utils

def test_resample(n=1e4):
    w = np.random.random(n)
    w = w / w.sum()
    u = np.random.uniform(size=1, low=0, high=1)
    r = list()
    for name in utils.opts:
        f = getattr(sys.modules['utils.'+name], 'resample')
        t = time.clock()
        r.append(f(w.copy(), u))
        print '%s:\t%.3f' % (name, time.clock() - t)
    assert all(all(x == r[0]) for x in r)

def main():
    test_resample()

if __name__ == "__main__":
    main()
