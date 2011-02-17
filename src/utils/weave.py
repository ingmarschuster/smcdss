#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

import scipy.weave as weave
import numpy as np

def resample(w, u):
    ''' Computes the particle indices by systematic resampling using scypy.weave.
        @deprecated method is not used any longer
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

def log_regr_rvs(Beta, u=None, gamma=None):
    d = Beta.shape[0]
    sample = int(u is not None)
    if u is not None:
        gamma = np.empty(d, dtype=bool)
        logu = np.log(u)
    else:
        logu = np.empty(1, dtype=bool)

    logp = np.zeros(1, dtype=float)
    code = \
    """
    double sum, logcprob;
    int i,j;
  
    for(i=0; i<d; i++){
    
        /* Compute log conditional probability that gamma(i) is one */
        sum = Beta(i,i);
        for(j=0; j<i; j++){ sum += Beta(i,j) * gamma(j); }
        logcprob = -log(1+exp(-sum));
        
        /* Generate the ith entry */
        if (sample) gamma(i) = (logu(i) < logcprob);
        
        /* Compute log conditional probability of whole gamma vector */
        logp += logcprob;        
        if (!gamma(i)) logp -= sum;
        
    }
    """
    weave.inline(code, ['d', 'logu', 'Beta', 'gamma', 'logp', 'sample'],
                 type_converters=weave.converters.blitz, compiler='gcc')
    return gamma, logp[0]
