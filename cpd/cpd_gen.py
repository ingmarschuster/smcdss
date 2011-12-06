#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Generator for change point detection problems.

@verbatim
USAGE:
        cpd <option>

OPTIONS:
        -k    number of data streams (integer)
        -t    number of observations (integer)
        -f    filename

@endverbatim
"""

"""
@namespace cpd.cpd_gen
$Author: christian.a.schafer@gmail.com $
$Rev: 144 $
$Date: 2011-05-12 19:12:23 +0200 (jeu., 12 mai 2011) $
@details
"""

import getopt
import sys
import os
import numpy
import cPickle as pickle
import cpd

def generate_cpd_problem(d, T, filename, n=None, loc=0.0, scale=1.0, shift=0.5):
    """
        Generates a change point detection problem.
        \param d number of streams
        \param T number of observations
        \param n number of affected streams
        \param loc mean of the normal
        \param scale variance of the normal
        \param shift shift of the mean
        \param filename filename of the cpd problem
    """

    # draw a uniform change point between 0 and n_obs+1
    cp_index = numpy.random.randint(low=int(0.25 * T) + 1, high=int(0.75 * T) + 1)
    cp = cp_index + 1
    
    # draw the number of streams affected by the change
    if n is None:n = numpy.random.randint(low=1, high=d + 1)

    print "Summary:"
    print "no observations  : %02d" % T
    print "change point     : %02d" % cp
    print "total streams    : %02d" % d
    print "affected streams : %02d" % n

    # data without change
    data = numpy.random.normal(loc=loc, scale=scale, size=(d, T))
    
    # n streams overwritten from the change point on...
    data[:n, cp_index:] = numpy.random.normal(loc=loc + shift, scale=scale, size=(n, T - cp_index))
    
    # choose a random permutation for the subset of affected streams
    perm = getPermutation(d)
    data = data[perm, :]
    
    args = {'loc'    : loc,
            'scale'  : scale,
            'shift'  : shift,
            'cp'     : cp,
            'subset' : numpy.array(perm) < n,
            'data'   : data,
            'd'      : data.shape[0],
            'T'      : data.shape[1]
            }
    
    # pickle args
    file = open(os.path.join(cpd.v['DATA_PATH'], filename + '.pickle'), 'w')
    pickle.dump(obj=args, file=file)
    file.close()

def load_cpd_problem(filename):
    file = open(os.path.join(cpd.v['DATA_PATH'], filename + '.pickle'), 'r')
    args = pickle.load(file)
    file.close()
    return args

def getRandomSubset(n, k):
    """
        Draw a uniformly random subset of the index set.
        @return subset
    """
    if k < 5:
        index = []
        while len(index) < k:
            i = numpy.random.randint(low=0, high=n)
            if not i in index: index.append(i)
    else:
        index = getPermutation()[:k]
    return index

def getPermutation(n):
        """
            Draw a random permutation of the index set [[0...n-1]].
            @return permutation
        """
        perm = range(n)
        for i in reversed(range(1, n)):
            # pick an element in p[:i+1] with which to exchange p[i]
            j = numpy.random.randint(low=0, high=i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

def main():
    
    # Parse command line options.
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:T:f:')
    except getopt.error, msg:
        print msg
        sys.exit(2)
        
    d, T, filename = None, None, None
    
    for o, a in opts:
        if o == '-d': d = int(a)
        if o == '-T': T = int(a)
        if o == '-f': filename = a
    
    for var in [d, T, filename]:
        if var is None:
            print "You need to specify dimension, type and filename."
            sys.exit(0)
    
    cpd.read_config()
    generate_cpd_problem(d=d, T=T, filename=filename)
    print "\nTest suite saved as '%s'." % filename

if __name__ == "__main__":
    main()
