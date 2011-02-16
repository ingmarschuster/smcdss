#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-12-10 19:39:02 +0100 (ven., 10 déc. 2010) $
    $Revision: 0 $
'''

import scipy.weave as weave
import numpy as np

def resample(w, u):
    '''
        Computes the particle indices by systematic resampling using scypy.weave.
        @param w array of weights
    '''
    code = \
    """
    int j = 0;
    double cumsum = weights(0);
    
    for(int k = 0; k < n; k++)
    {
        while(cumsum < u)
        {
        j++;
        cumsum += weights(j);
        }
        i(k) = j;
        u = u + 1.;
    }
    """
    n = w.shape[0]
    u = float(u)
    weights = n * w
    i = np.zeros(n, dtype='int')
    weave.inline(code, ['u', 'n', 'weights', 'i'], type_converters=weave.converters.blitz, compiler='gcc')
    return i
