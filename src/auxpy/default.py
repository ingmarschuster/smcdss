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
from binary import LogisticRegrBinary, ProductBinary, HiddenNormalBinary

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
    n_particles=10000,

    # minimum distance of mean from the boundaries of [0,1]
    min_d=1e-04,

    # epsilon
    eps=0.05,

    # delta
    delta=0.1,
    
    # tau
    tau=0.7

)

dicCE = dict(

    # number of particles 
    n_particles=1000,

    # model for binary data with dependencies
    model=LogisticRegrBinary,
    #model=ProductBinary,
    #model=HiddenNormalBinary,

    # elite fraction used to estimate the mean
    elite=0.5,

    # lag in mean update
    lag=0.2,

    # minimum distance of mean from the boundaries of [0,1]
    min_d=1e-04,

    # epsilon
    eps=0.05,

    # delta
    delta=0.1

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
    boxplot=0.8,

    # print titles
    title=False,

    # colored graph
    color=False,

    # default path to evaluation directory 
    eval_path='/home/cschafer/Documents/smcdss/data/testruns',

    # outer margins (see R)
    outer_margin=[40, 4, 5, 4],

    # inner margins (see R)    
    inner_margin=[0, 0, 0, 0],

    # font family (see R)      
    font_family='serif',

    # lines below main title
    title_line=1

)
