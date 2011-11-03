#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Change-point detection statistics.
"""

"""
@namespace cpd.stats
$Author: christian.a.schafer@gmail.com $
$Rev: 144 $
$Date: 2011-05-12 19:12:23 +0200 (jeu., 12 mai 2011) $
@details
"""

import os
import numpy
import subprocess
import scipy.stats as stats
import utils
import cpd


def main():
       
    # generate test problem
    cpd.cpd_gen.generate_cpd_problem(d=15, T=100, filename='test', shift=0.5, n=5)
    args = cpd.cpd_gen.load_cpd_problem('test')
    
    # params
    alpha = 2 ** (args['d'] + 1)
    lambda_ = 0.05
    ticks = 10
    
    # compute statistic sequentially    
    for stat in [BayesStat(Model(args), lambda_=lambda_), FSumStat(Model(args), alpha=alpha), FSingleStat(Model(args), alpha=alpha), FMultiStat(Model(args), alpha=alpha)]:
        process_stat(stat, args, ticks=ticks)


def process_stat(stat, args, ticks=5):
    """
        Computes the change-point statistics for a number of sampling points and
        plots the evolution.
        \param stat statistic object
        \param ticks number of sampling points
    """

    # unpack arguments
    d, T, cp, subset, data = [args[k] for k in ['d', 'T', 'cp', 'subset', 'data']]
    
    # calculate uniform sampling points
    points = range(T / ticks + 1, T + 1, T / ticks + 1) + [T]

    # initialize containers    
    t1, changepoint_estimator, time_marginals, subset_estimator = 0, [], [], []
    for t2 in points:
        
        # append data
        stat.append(data[:, t1:t2])
        
        # store statistics
        changepoint_estimator += [stat.changepoint_estimator()]
        subset_estimator += list(stat.subset_estimator())
        time_marginals += list(stat.time_marginals())
        t1 = t2
   
    title = """change-point detection using "%s" statistic for %d observations of %d
               standard Gaussian data streams with change in mean by %.1f""" % (stat.name, T, d, stat.model.shift)
    cpd.v['RUN_FOLDER'] = os.path.join(cpd.v['RUN_PATH'], 'test')
    
    # open R-template
    f = open(os.path.join(cpd.v['SYS_ROOT'], 'src', 'cpd', 'cpd_plot.R') , 'r')
    R_script = f.read() % {
            'T'          : T,
            'd'          : d,
            'points'     : ', '.join(['%d' % x for x in points]),
            'cp'         : cp,
            'cp_est'     : ', '.join(['%.5f' % x for x in changepoint_estimator]),
            'subset'     : ', '.join(['%d' % x for x in subset]),
            'subset_est' : ', '.join(['%.5f' % x for x in subset_estimator]),
            'tm_data'    : ', '.join(['%.5f' % x for x in time_marginals]),
            'title'      : title.replace('\n', '').replace('  ', ''),
            'stat'       : stat.name,
            'file'       : os.path.join(cpd.v['RUN_FOLDER'], 'plot%s.pdf' % stat.name)
            }
    f.close()
    
    # copy plot.R to its run folder
    f = open(os.path.join(cpd.v['RUN_FOLDER'], 'plot%s.R' % stat.name), 'w')
    f.write(R_script)
    f.close()

    # execute R-script
    subprocess.Popen([cpd.v['SYS_R'], 'CMD', 'BATCH', '--vanilla',
                      os.path.join(cpd.v['RUN_FOLDER'], 'plot%s.R' % stat.name),
                      os.path.join(cpd.v['RUN_FOLDER'], 'plot%s.Rout' % stat.name)]).wait()


def calibrate_frate(alpha, d, t):
    """
        Computes a failure rate lambda_ such at time t the prior probability of
        no change corresponds to the threshold level alpha in the Frequentist
        framework.
    """
    lambda_ = 10.0 / float(t)
    c = (2 ** d - 1) / float(alpha)
    
    # solve (2^d-1)/(lambda(1+lambda)^t)=alpha for lambda via Newton iterations
    for i in xrange(50):
        lambda_last = lambda_
        lambda_ -= (lambda_ * (1 + lambda_) ** t - c) / ((1 + lambda_) ** (t - 1) * (1 + lambda_ + t * lambda_))
        if numpy.abs(lambda_ - lambda_last) < 1e-8: break
    return lambda_


#------------------------------------------------------------------------------ 


class Model(object):
    
    BOUND = 1e-50
    
    def __init__(self, args):
        ## parameters of the stream data
        self.d, self.loc, self.scale, self.shift = [args[k] for k in ['d', 'loc', 'scale', 'shift']]
    
    def log_likelihood(self, y):
        """
            \param y numpy array
            \return z_{k,*} = log [ g_k(y_{k,*}) / f_k(y_{k,*}) ]
        """
        g = stats.norm.pdf(y, loc=self.loc + self.shift, scale=self.scale)
        f = stats.norm.pdf(y, loc=self.loc, scale=self.scale)
        return numpy.log(g + self.BOUND) - numpy.log(f + self.BOUND)


class Stat(object):
    """
        change-point statistic interface class
    """
    
    name = 'stat'
    
    def __init__(self, model):
        """
            \param model a data model that allows for evaluation of the
            unnormalized log-likelihood of each data stream
        """
        
        ## number of observations
        self.t = 0
        
        ## number of streams
        self.d = model.d

        ## underlying data model
        self.model = model
        
        ## system of all non-empty subsets
        if not hasattr(self, 'B'): self.B = self.init_system(self.d)

        ## matrix of log ratios
        self.Z = numpy.empty(shape=(self.d, self.t), dtype=float) 

    def append(self, y):
        """
            \param y numpy array of shape=(d,n)   
        """
        for y in y.T: self.append_single(y)
        
    def append_single(self, y):
        """
            \param y numpy array of shape=(d,)
        """
        # unnormalized log-likelihood
        z = self.model.log_likelihood(y)[:, numpy.newaxis]
        # append
        self.Z = numpy.concatenate((self.Z, z), axis=1)
        # update time
        self.t += 1
        # update statistics
        self.update(z)

    def log_v_t_gamma(self, z, gamma=None):
        """
            \param y_t numpy array of shape=(d,)
            \return exp(gamma^t Z_t)
        """
        if gamma is None: gamma = self.B
        return numpy.dot(gamma, z)

    def update(self, y_s):
        """
            update statistics
        """
        pass
    
    def time_marginals(self):
        """
            \return statistics for all times averaged over all subsets
        """
        return numpy.zeros(self.t, dtype=float)

    def time_subsets(self):
        """
            \return statistics for all times and all subsets
        """
        times_streams = numpy.array([self.Z[:, s:self.t].sum(axis=1) for s in range(self.t)]).T
        return numpy.dot(self.B, times_streams)

    def changepoint_estimator(self):
        """
            \return change-point estimate
        """
        return 0.0

    def subset_estimator(self):
        """
            \return affected subset estimate
        """
        return numpy.zeros(self.d, dtype=float)
    
    def change_estimator(self):
        """
            \return event of change estimate
        """
        return False

    @staticmethod
    def init_system(d):
        """
            \return system of all non-empty subsets.
        """
        B = numpy.empty(shape=(2 ** d - 1, d), dtype=bool)
        for dec in xrange(2 ** d - 1): B[dec, :] = (utils.format.dec2bin(dec + 1, d))
        return B


class BayesStat(Stat):
    """
        Bayesian change-point statistic using a geometric prior distribution on
        the change-point and a uniform on the affected subset
    """
    
    name = 'Bayesian'
    
    ## posterior probability of change event necessary to raise an alarm
    threshold = 0.6
    
    def __init__(self, model, lambda_):
        """
            \param model a data model that allows for evaluation of the
            unnormalized log-likelihood of each data stream
            \param lambda_ the failure rate of the change event. 1/(1+lambda_)
            is the parameter of the geometric prior.
        """
        Stat.__init__(self, model)
        
        ## failure rate of the observed process
        self.lambda_ = lambda_
        
        ## unnormalized Bayesian change-point estimate 
        self.tau_gamma = numpy.zeros(shape=(2 ** self.d - 1, 1), dtype=float)
        
        ## unnormalized probability
        self.v_gamma = numpy.zeros(shape=(2 ** self.d - 1, 1), dtype=float)

        ## unnormalized probability of change
        self.v = 0.0
        
        ## unnormalized probability of no change
        self.n = 1.0

        ## constant for the probability of no change
        self.n_constant = (2 ** self.d - 1.0) / float(lambda_)

    def update(self, z):
        """
            update statistics
        """
        
        v_t_gamma = numpy.exp(self.log_v_t_gamma(z))
        
        # update v_gamma statistic
        self.v_gamma = v_t_gamma * (self.v_gamma + (1.0 + self.lambda_) ** -self.t)

        # update tau_gamma statistic
        self.tau_gamma = v_t_gamma * (self.tau_gamma + self.t * (1.0 + self.lambda_) ** -self.t)

        # unnormalized probability of change
        self.v = self.v_gamma.sum()
        
        # unnormalized probability of no change
        self.n = self.n_constant * (1.0 + self.lambda_) ** -self.t


    def time_marginals(self):
        """
            \return statistics for all times averaged over all subsets
        """
        # log posterior for all times and subsets
        times_subsets = self.time_subsets() - numpy.mgrid[1:2 ** self.d, 1:self.t + 1][1] * numpy.log(1.0 + self.lambda_)
        
        # unnormalized marginal probabilities
        log_max = times_subsets.max()
        probs = numpy.exp(times_subsets - log_max).sum(axis=0) * numpy.exp(log_max)
        
        # add probability of no change
        probs = numpy.append(probs, [self.n])
        
        # normalized marginal probabilities
        return probs / (self.n + self.v)

    def changepoint_estimator(self):
        """
            \return change-point estimate
        """
        if not self.change_estimator(): return self.t + 1
        else: return self.tau_gamma.sum() / self.v

    def subset_estimator(self):
        """
            \return affected subset estimate
        """
        if not self.change_estimator(): return numpy.zeros(self.d, dtype=float)
        else: return numpy.dot(self.B.T, self.v_gamma) / self.v
    
    def change_estimator(self):
        """
            \return event of change estimate
        """
        pp = self.v / (self.v + self.n)
        return pp > self.threshold
    

class FMultiStat(Stat):
    """
        Frequentist change-point statistic using multiple repeated likelihood-
        ratio testing on all subsets of data streams
    """
    name = 'FMultiple'
    
    def __init__(self, model, alpha):
        """
            \param model a data model that allows for evaluation of the
            unnormalized log-likelihood of each data stream
            \param alpha the threshold for the g-statistic 
        """
        Stat.__init__(self, model)
        
        ## threshold
        self.alpha = alpha
        
        ## unnormalized MAPs per subset
        self.w_gamma = numpy.zeros(shape=(self.B.shape[0], 1), dtype=float)
       
        ## MAP over all subsets
        self.g = 0.0
        
        ## index that yields the MAP in w_gamma
        self.argmax_g = 0

    def update(self, z):
        """
            update statistics
        """
        # update w_gamma statistic
        self.w_gamma = numpy.maximum(self.w_gamma + self.log_v_t_gamma(z), 0.0)
        self.argmax_g = numpy.argmax(self.w_gamma)
        self.g = self.w_gamma[self.argmax_g, 0]
    
    def time_marginals(self):
        """
            \return statistics for all past times averaged over all subsets
        """
        # maximize over all subsets
        times = self.time_subsets().max(axis=0)
        
        # add log probability of no change
        times = numpy.append(times, [0.0])
       
        # unnormalized marginals
        probs = numpy.exp(times - times.max())

        # normalized marginals
        return probs / probs.sum()

    def changepoint_estimator(self):
        """
            \return change-point estimate
        """
        if not self.change_estimator(): return self.t + 1

        # maximize over subsets
        times = self.time_subsets().max(axis=0)
        
        return numpy.argmax(times) + 1

    def subset_estimator(self):
        """
            \return affected subset estimate
        """
        if not self.change_estimator(): return numpy.zeros(self.d, dtype=float)
        else: return self.B[self.argmax_g]
    
    def change_estimator(self):
        """
            \return event of change estimate
        """
        return self.g > numpy.log(self.alpha)
   

class FSingleStat(FMultiStat):
    """
        Frequentist change-point statistic using multiple repeated likelihood-
        ratio testing on single data streams
    """
    name = 'FSingle'
    
    def __init__(self, model, alpha):
        """
            \param model a data model that allows for evaluation of the
            unnormalized log-likelihood of each data stream
            \param alpha the threshold for the g-statistic 
        """

        ## system of all single streams
        self.B = numpy.eye(model.d, dtype=bool)

        FMultiStat.__init__(self, model, alpha)


class FSumStat(FSingleStat):
    """
        Frequentist change-point statistic using multiple repeated likelihood-
        ratio testing on the sum of the single stream statistics
    """
    name = 'FSum'
    
    def __init__(self, model, alpha):
        """
            \param model a data model that allows for evaluation of the
            unnormalized log-likelihood of each data stream
            \param alpha the threshold for the g-statistic 
        """
        FSingleStat.__init__(self, model, alpha)

    def update(self, z):
        """
            update statistics
        """
        # update w_gamma statistic
        self.w_gamma = numpy.maximum(self.w_gamma + self.log_v_t_gamma(z), 0.0)
        self.g = self.w_gamma.sum()
    
    def time_marginals(self):
        """
            \return statistics for all past times averaged over all subsets
        """ 
        # maximize over all subsets
        times = self.time_subsets().sum(axis=0)
        
        # add log probability of no change
        times = numpy.append(times, [0.0])
       
        # unnormalized marginals
        probs = numpy.exp(times - times.max())

        # normalized marginals
        return probs / probs.sum()

    def changepoint_estimator(self):
        """
            \return change-point estimate
        """
        if not self.change_estimator(): return self.t + 1
        
        # sum over all single streams
        times = self.time_subsets().sum(axis=0)
        return numpy.argmax(times) + 1

    def subset_estimator(self):
        """
            \return affected subset estimate
        """
        if not self.change_estimator(): return numpy.zeros(self.d, dtype=float)
        else: return self.w_gamma / self.w_gamma.max()
    
    def change_estimator(self):
        """
            \return event of change estimate
        """
        return self.g > numpy.log(self.alpha)

#------------------------------------------------------------------------------ 


if __name__ == "__main__":
    cpd.read_config()
    main()

















'''
#------------------------------------------------------------------------------ 


def mei_bf(filename, ticks=4):
    
    # load problem
    file = open(os.path.join(cpd.v['DATA_PATH'], filename + '.pickle'), 'r')
    L = pickle.load(file)
    file.close()
    
    # unpack arguments
    cp, gamma_affected, data = L['change_point'], L['affected'], L['data']
    
    # compute log ratio of likelihoods
    g = stats.norm.pdf(data, loc=L['loc'] + L['shift'], scale=L['scale'])
    f = stats.norm.pdf(data, loc=L['loc'], scale=L['scale'])
    log_ratio = numpy.log(g / f)
    
    # dimension of the problem
    k, t = L['data'].shape
    
    # time samples
    s = range(t / ticks + 1, t + 1, t / ticks + 1) + [t]
    
    # indices of the affected streams
    st_af = numpy.where(gamma_affected)[0] + 1
    
    # initialize empty lists
    cp_est, st_data, cp_data = [[] for x in xrange(3)]
    for r in s:
    
        # marginal, normalized likelihoods of change points
        cp_data_r = list(exp_normalize(V_max(r, log_ratio)))
        cp_data += cp_data_r
        
        # change point estimates and most likely subset
        cp_est_r, st_data_r = G_max(r, log_ratio)
        
        st_data += list(st_data_r)
        cp_est += [cp_est_r]
    
    title = 'changepoint detection for %d observations of %d standard Gaussian data streams with change in mean by %.1f' % (t, k, L['shift'])
   
    plot(cp=cp, cp_est=cp_est, t=t, s=s, k=k, st_af=st_af, st_data=st_data, cp_data=cp_data, title=title)

def G_max(s, log_ratio):
    """
        Maximum likelihood over time and all subsets
        \return estimator, argmax gamma
    """
    n_streams = log_ratio.shape[0]
    # initialize with non-empty subset
    gamma_argmax = utils.format.dec2bin(1, n_streams)
    G_time_argmax, G_max_value = W_max(s, gamma_argmax, log_ratio)
    
    # loop over all non-empty subsets
    for dec in xrange(2, 2 ** n_streams):
        gamma = utils.format.dec2bin(dec, n_streams)
        W_time_argmax, W_max_value = W_max(s, gamma, log_ratio)
        # check for larger value
        if W_max_value > G_max_value:
            G_time_argmax, G_max_value = W_time_argmax, W_max_value
            gamma_argmax = gamma

    # convert index to time-line value
    G_time_argmax += 1
    return G_time_argmax, gamma_argmax

def W_max(s, gamma, log_ratio, V_all_arr=None):
    """
        Likelihood for a subset gamma maximized over all times <= s
        \return estimator given gamma, max value
    """
    if V_all_arr is None: V_all_arr = V_all(s, gamma, log_ratio)
    W_time_argmax = numpy.argmax(V_all_arr)
    W_max_value = V_all_arr[W_time_argmax]
    return W_time_argmax, W_max_value

def V_max(s, log_ratio):
    """
        Likelihood for all times maximized over all subsets gamma
        \return estimator given gamma, max value
    """
    n_streams = log_ratio.shape[0]
    
    # initialize with non-empty subset
    likelihood = V_all(s, utils.format.dec2bin(1, n_streams), log_ratio)
    
    # loop over all non-empty subsets
    for dec in xrange(2, 2 ** n_streams):
        gamma = utils.format.dec2bin(dec, n_streams)
        likelihood = numpy.maximum(V_all(s, gamma, log_ratio), likelihood)
    return likelihood

def V_all(s, gamma, log_ratio):
    return numpy.array([V(cp=cp, s=s, gamma=gamma, log_ratio=log_ratio) for cp in range(s)])

def V(cp, s, gamma, log_ratio):
    # sum over all log ratios from change point to s
    log_ratio_sum = log_ratio[:, cp:s].sum(axis=1)
    # sum over all stream in the subset gamma
    return numpy.dot(log_ratio_sum, gamma)
'''

'''

class ChangePointMonitor(object):

    def __init__(self, **L):
        
        self.verbose = False
       
        # unpack parameters
        final_t, self.alpha, self.d, self.loc, self.scale, self.shift = [L[k] for k in ['final_t', 'alpha', 'd', 'loc', 'scale', 'shift']]
        
        # non-empty subsets
        self.B = list()
        for dec in xrange(1, 2 ** self.d):
            self.B.append(utils.format.dec2bin(dec, self.d))
        self.B = numpy.array(self.B)
        
        # current time
        self.t = 0
       
        # log ratios until time t
        self.log_ratio = None
        
        # estimate reasonable value for the failure rate
        frate = 0.1 # self.estimate_frate(t=final_t * 0.75)

        # constants
        self.c1 = numpy.log(1 + frate)
        self.c2 = numpy.log(2 ** self.d - 1) - numpy.log(frate)

    def estimate_frate(self, t):
        """
            Computes a failure rate lambda such at time t the prior probability
            of no change corresponds to the threshold level alpha in the
            frequentist framework.
        """
        x = 10.0 / float(t)
        c = (2 ** self.d - 1) / float(self.alpha)
        
        # solve (2^d-1)/(lambda(1+lambda)^t)=alpha for lambda via Newton iterations
        for i in xrange(50):
            z = x
            x -= (x * (1 + x) ** t - c) / ((1 + x) ** (t - 1) * (1 + x + t * x))
            if numpy.abs(x - z) < 1e-8: break
        return x
        
    def append(self, data):
        """
            Adds an observation.
            \param data vector of observations at all streams
        """
        # compute log likelihoods
        g = stats.norm.pdf(data, loc=self.loc + self.shift, scale=self.scale)
        f = stats.norm.pdf(data, loc=self.loc, scale=self.scale)
        if self.verbose:
            print utils.format.format(g, 'g')
            print utils.format.format(f, 'f')
        
        # append log likelihood ratios
        log_ratio = numpy.log(g + 1e-50) - numpy.log(f + 1e-50)
        if self.log_ratio is None: self.log_ratio = log_ratio
        else: self.log_ratio = numpy.concatenate((self.log_ratio, log_ratio), axis=1)
        self.t += data.shape[1]

        self.update_single_stream()
        self.update_multiple_stream()
        self.update_bayes()
        

    def update_multiple_stream(self):
        """
            Updates the maximum likelihood ratio over all subsets of streams.
        """
        
        # log likelihood ratio sums from the changepoint to t for all streams (k x t) -- these could be updated!
        V_per_stream = numpy.array([self.log_ratio[:, cp:self.t].sum(axis=1) for cp in range(self.t + 1)]).T

        # log likelihood ratio sums for all non-empty subsets (2**k-1 x k) * (k x t) = (2**k-1 x t) -- this could be updated!
        self.V_per_subset = numpy.dot(self.B, V_per_stream)

        # maximum over subsets (likelihood ratios)
        self.V_max_subset = numpy.amax(self.V_per_subset, axis=0)
        self.V_max_subset = exp_normalize(self.V_max_subset)
        
        if self.verbose:
            print utils.format.format(self.V_per_subset, 'V per subset')
            print utils.format.format(self.V_max_stream, 'max streams V')
            print utils.format.format(self.V_max_subset, 'normalized max subset V')

        # maximum over changepoints (most likely affected subset)
        W_index = numpy.argmax(self.V_max_subset) # argmax over time
        if W_index == self.t:
            self.S_subset = numpy.zeros(self.d, dtype=float)
        else:    
            S_index = numpy.argmax(self.V_per_subset[:, W_index], axis=0) # argmax over subsets
            self.S_subset = self.B[S_index]

        # maximum over subsets and changepoints
        G_max_subset_index = numpy.argmax(self.V_max_subset)
        if self.V_max_subset[G_max_subset_index] > self.alpha * self.V_max_subset[-1]:
            self.G_max_subset = G_max_subset_index + 1
        else:
            self.G_max_subset = self.t + 1
            self.S_subset = numpy.zeros(self.d, dtype=float)

    def update_single_stream(self):
        """
            Updates the maximum likelihood ratio over all single streams.
        """
        
        # log likelihood ratio sums from the changepoint to t for all streams (k x t) -- these could be updated!
        V_per_stream = numpy.array([self.log_ratio[:, cp:self.t].sum(axis=1) for cp in range(self.t + 1)]).T
        
        # maximum over single streams (likelihood ratios)
        self.V_max_stream = numpy.amax(V_per_stream, axis=0)
        self.V_max_stream = exp_normalize(self.V_max_stream)
        
        # maximum over single streams and changepoints
        G_max_stream_index = numpy.argmax(self.V_max_stream)
        if self.V_max_stream[G_max_stream_index] > self.alpha * self.V_max_stream[-1]: self.G_max_stream = G_max_stream_index + 1
        else: self.G_max_stream = self.t + 1
        
        if self.verbose:
            print utils.format.format(V_per_stream, 'v per streams')            
            print utils.format.format(self.V_max_stream, 'normalized max streams V')
        
        # maximum over changepoints (most likely affected subset)
        W_index = numpy.argmax(self.V_max_stream) # argmax over time
        self.S_stream = numpy.zeros(self.d, dtype=float)
        if W_index < self.t:   
            S_index = numpy.argmax(V_per_stream[:, W_index], axis=0) # argmax over subsets
            self.S_stream[S_index] = True

    def update_bayes(self):
        """
            Updates the Bayesian posterior probability over all subsets of streams.
        """
        
        # log likelihood ratio sums for all non-empty subsets (2**k-1 x k) * (k x t) -- this could be updated!
        self.bayes_per_stream = numpy.array([self.log_ratio[:, cp:self.t].sum(axis=1) for cp in range(self.t + 1)]).T
        
        self.bayes_all_subsets = numpy.dot(self.B, self.bayes_per_stream)
        self.bayes_all_subsets[:, :-1] -= numpy.mgrid[1:2 ** self.d, 1:self.t + 1][1] * self.c1
        self.bayes_all_subsets[:, -1] = self.c2 - self.t * self.c1 # posterior probability of no change
        
        self.bayes_all_subsets = exp_normalize(self.bayes_all_subsets)
        
        self.V_bayes = self.bayes_all_subsets.sum(axis=0)
        self.S_bayes = numpy.dot(self.B.T, self.bayes_all_subsets[:, :-1]).sum(axis=1)
        
        if self.verbose:
            print utils.format.format(self.V_bayes, 'marginal time')
            print utils.format.format(self.S_bayes, 'marginals subsets')

        # compute Bayes estimator
        if  numpy.argmax(self.V_bayes) < self.t:
            self.G_bayes = 0.0
            for i, p in enumerate(self.V_bayes):
                self.G_bayes += p * (i + 1)
        else:
            self.G_bayes = float(self.t + 1)
            self.S_bayes = numpy.zeros(self.d, dtype=float)
            
def exp_normalize(w):
    """
        \param w vector
        \return exp(w)/exp(w).sum() 
    """
    w = w - w.max()
    w = numpy.exp(w)
    return w / w.sum()
'''

'''
def mei_seq(filename, alpha=1.0, ticks=5):
    """
        Computes the changepoint statistics for the problem specified in
        the data file <filename>.
        \param filename data file
        \param ticks number of sampling points
    """

    # load problem
    file = open(os.path.join(cpd.v['DATA_PATH'], filename + '.pickle'), 'r')
    L = pickle.load(file)
    file.close()

    # unpack arguments
    cp, subset_affected, data = [L[k] for k in ['change_point', 'affected', 'data']]
    
    # indices of the affected streams
    st_af = numpy.where(subset_affected)[0] + 1
      
    # time samples
    k, final_t = data.shape
    cpm = ChangePointMonitor(final_t=final_t, alpha=alpha, **L)
    
    # calculate uniform sampling points
    sample_points = range(final_t / ticks + 1, final_t + 1, final_t / ticks + 1) + [final_t]   

    # initialize containers    
    t_0, cp_est_bayes, cp_data_bayes, st_data_bayes, cp_est_subset, cp_data_subset, st_data_subset, cp_est_stream, cp_data_stream, st_data_stream = \
        0, [], [], [], [], [], [], [], [], []
    for t in sample_points:
        # propagate changepoint-monitor
        cpm.append(data[:, t_0:t])
        # retrieve statistics
        cp_est_bayes += [cpm.G_bayes]
        cp_data_bayes += list(cpm.V_bayes)
        st_data_bayes += list(cpm.S_bayes)
        cp_est_subset += [cpm.G_max_subset]
        cp_data_subset += list(cpm.V_max_subset)
        st_data_subset += list(cpm.S_subset)
        cp_est_stream += [cpm.G_max_stream]
        cp_data_stream += list(cpm.V_max_stream)
        st_data_stream += list(cpm.S_stream)
        t_0 = t
   
    # plot the statistics
    for cp_est, cp_data, st_data, stat in [(cp_est_subset, cp_data_subset, st_data_subset, 'subset'),
                                           (cp_est_stream, cp_data_stream, st_data_stream, 'stream'),
                                           (cp_est_bayes, cp_data_bayes, st_data_bayes, 'Bayes')]:
        title = 'changepoint detection using "%s" statistic for %d observations of %d standard Gaussian data streams with change in mean by %.1f' % (stat, final_t, k, cpm.shift)
        plot(cp=cp, cp_est=cp_est, t=final_t, s=sample_points, k=k, st_af=st_af, st_data=st_data, cp_data=cp_data, title=title, stat=stat)
'''
