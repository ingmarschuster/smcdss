#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import time, sys
import numpy as np

import utils
import data
from binary import logistic_model, normal_model

def test_log_regr(d=4, n=100):

    q = logistic_model.LogisticBinary.random(d)
    s = data.data()
    s.sample(q=q, size=n)
    b = logistic_model.LogisticBinary.from_data(s)
    print s
    print b.marginals()


def test_log_regr_rvs(d=10, n=5000):

    b = logistic_model.LogisticBinary.random(d)
    r = list()
    print 'sampling'
    for name in utils.opts:
        np.random.seed(0)
        f = getattr(sys.modules['utils.' + name], 'log_regr_rvs')
        t = time.clock()
        s = list()
        for i in range(n):
            u = np.random.random(d)
            s.append(f(b.Beta, u=u))
        r.append(s)
        print '%s:\t%.3f' % (name, time.clock() - t)
    print '\nevaluating'
    for name in utils.opts:
        f = getattr(sys.modules['utils.' + name], 'log_regr_rvs')
        t = time.clock()
        for i in range(n):
            assert abs(f(b.Beta, gamma=r[0][i][0])[1] - r[0][i][1]) < 1e-8
        print '%s:\t%.3f' % (name, time.clock() - t)

    if n < 5:
        for i in range(n):
            for x in r:
                print x[i][0], x[i][1]

def test_resample(n=5e5):
    w = np.random.random(n)
    w = w / w.sum()
    u = np.random.uniform(size=1, low=0, high=1)
    r = list()
    for name in utils.opts:
        f = getattr(sys.modules['utils.' + name], 'resample')
        t = time.clock()
        r.append(f(w.copy(), u))
        print '%s:\t%.3f' % (name, time.clock() - t)
    assert all(all(x == r[0]) for x in r)

def main():
    #test_resample()
    #test_log_regr_rvs()
    test_log_regr()

if __name__ == "__main__":
    main()
