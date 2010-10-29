#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    This module holds dictionaries with default settings.

    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from numpy import inf

dicMC = dict(

    # Markov kernel
    kernel='MH',

    # max running time in minutes
    max_time=30.0,

    # max iterations
    max_iter=inf,

)

dicSMC = dict(

    # number of particles
    n_particles=20000,

    # number of bridge steps
    n_bridge= -1,

    # kappa ?
    kappa=0,

)

dicCE = dict(

    # number of particles 
    n_particles=10000,

    # model for binary data with dependencies
    dep_model='logistic-regression-binary',

    # elite fraction used to estimate the mean
    elite_mean=0.02,

    # elite fraction used to estimate the correlation matrix
    # or the logistic regression coefficients
    elite_corr=0.2,

    # lag in mean update
    lag_mean=0.3,

    # lag in correlation update
    lag_corr=0.2,

    # minimum distance of mean from the boundaries of [0,1]
    eps=0.02,

)

dicSA = dict(

    # number of steps
    n_steps=300000

)

dicTest = dict(

    # write result into an output file
    output_file=False,

    # write extensive information to stdout
    verbose=False,
    
    # number of runs to be performed
    runs=200

)

dicData = dict(

    # type of model used as posterior distribution of the variable selection problem:
    # a conjugate Hierarchical Bayesian setup (hb) or the Bayesian Information Criterion (bic)
    model_type='hb',
    
    # data set to perform the variable selection on
    dataset='boston',
    
    # default path to data directory 
    data_path='/home/cschafer/Documents/smcdss/data/datasets',

)

dicEval = dict(

    # percentage of data to be contained in the box
    boxplot = 0.8,
    
    # print titles
    title = False,

    # colored graph
    color = False,
    
    # default path to evaluation directory 
    eval_path = '/home/cschafer/Documents/smcdss/data/testruns',
    
    # outer margins (see R)
    outer_margin = [40, 4, 5, 4],

    # inner margins (see R)    
    inner_margin = [0, 0, 0, 0],

    # font family (see R)      
    font_family = 'serif',
    
    # lines below main title
    title_line = 1

)
