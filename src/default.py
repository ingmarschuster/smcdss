#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    This module hold a dictionary with default settings.

    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from numpy import inf
from binary import *
from algos import *
from os.path import normcase

param = dict(




    #---------------------------------------------------- Sequential Monte Carlo

    # number of cpus
    smc_ncpus='autodetect',

    # number of particles
    smc_n=20000,

    # epsilon
    smc_eps=0.02,

    # delta
    smc_delta=0.075,

    # tau
    smc_tau=0.65,

    # minimum distance of mean from the boundaries of [0,1]
    smc_xi=1e-12,

    # binary model used in mh kernel
    smc_binary_model=LogisticBinary,


    #------------------------------------------------------------- Cross-Entropy

    # number of particles 
    ce_n=5000,

    # model for binary data with dependencies
    ce_model=LogisticBinary,

    # elite fraction used to estimate the mean
    ce_elite=0.5,

    # lag in mean update
    ce_lag=0.2,

    # minimum distance of mean from the boundaries of [0,1]
    ce_xi=1e-04,

    # epsilon
    ce_eps=0.075,

    # delta
    ce_delta=0.1,




    #-------------------------------------------------- Markov chain Monte Carlo

    # Markov kernel
    mcmc_kernel=SymmetricMetropolisHastings,

    # max iterations
    mcmc_max_evals=2e6,

    mcmc_q=1,


    #------------------------------------------------------- Simulated annealing

    # Markov kernel
    sa_kernel='mh',

    # max running time in minutes
    sa_max_time=30.0,

    # max iterations
    sa_max_iter=inf,




    #----------------------------------------------------------- Data processing

    # data set to perform the variable selection on
    data_set='boston',

    # number of covariates
    data_n_covariates=inf,

    # default path to data directory 
    data_path=normcase('data/datasets'),

    # a conjugate Hierarchical Bayesian setup (hb) or the Bayesian Information Criterion (bic)
    posterior_type='hb',




    #------------------------------------------------------------------- Testing

    # default path to test run directory 
    test_algo=smc,

    # default path to test run directory 
    test_path=normcase('data/testruns'),

    # write result into an output file
    test_output=False,

    # write extensive information to stdout
    test_verbose=False,

    # number of runs to be performed
    test_runs=200,




    #---------------------------------------------------------------- Evaluation

    # layout on A4 landscape
    eval_a4=True,

    # height of graph
    eval_height=4,

    # width of graph
    eval_width=12,

    # show names of covariates
    eval_names=False,
    
    # title
    eval_title=False,

    # percentage of data to be contained in the box
    eval_boxplot=0.8,

    # colored graph
    eval_color=['azure1', 'black', 'white', 'white', 'black'],

    # default path to evaluation directory 
    eval_path=normcase('data/evaluations'),

    # outer margins (bottom, left, top, right) 
    eval_outer_margin=[0, 0, 0, 0],

    # inner margins (bottom, left, top, right) 
    #eval_inner_margin=[5, 2, 5, 0],
    eval_inner_margin=[2, 2, 0.5, 0],

    # font family (see R)      
    eval_font_family='serif',

    # font family (see R)      
    eval_font_cex=1,

    # lines below main title
    eval_title_line=1

)
