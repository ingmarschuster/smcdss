#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Sequential Monte Carlo on binary spaces.
"""

"""
@namespace ibs.smc
$Author$
$Rev$
$Date$
@details
"""

import time
import datetime
import sys
import operator
import numpy
import ibs
import resample
import utils

class smc():
    """ Auxiliary class. """
    header = ['TYPE', 'NO_EVALS', 'TIME']
    @staticmethod
    def run(v): return integrate_smc(v)

def integrate_smc(param):

    print '\nRunning SMC using %d particles using %s and %s...' % \
        (param['SMC_N_PARTICLES'], param['SMC_BINARY_MODEL'].name, param['SMC_CONDITIONING'])
    ps = ParticleSystem(param)

    # run sequential MC scheme
    while ps.rho < 1.0:

        ps.fit_proposal()
        ps.condition()
        ps.reweight()

    sys.stdout.write('\rsmc completed in %s.\n' % (str(datetime.timedelta(seconds=time.time() - ps.start))))

    return ps.get_csv()


class ParticleSystem(object):

    def __init__(self, v):
        """
            Constructor.
            @param param parameters
            @param verbose verbose
        """

        if v['SMC_CONDITIONING'] == 'augment-resample': self.condition = self.augment_resample
        if v['SMC_CONDITIONING'] == 'augment-resample-unique': self.condition = self.augment_resample_unique
        if v['SMC_CONDITIONING'] == 'resample-move': self.condition = self.resample_move
        self.reweight = self.reweight_ess

        self.verbose = v['RUN_VERBOSE']
        if self.verbose: sys.stdout.write('...\n\n')

        self.start = time.time()

        if 'cython' in utils.opts: self._resample = utils.cython.resample
        else: self._resample = utils.python.resample

        ## target function
        self.f = v['f']
        self.job_server = v['JOB_SERVER']

        ## proposal model
        self.prop = v['SMC_BINARY_MODEL'].uniform(self.f.d)

        ## dimension of target function
        self.d = self.f.d
        ## number of particles
        self.n = v['SMC_N_PARTICLES']

        ## array of log weights
        self.log_W = numpy.zeros(self.n, dtype=float)
        ## array of log evaluation of the proposal model
        self.log_prop = numpy.empty(self.n, dtype=float)
        ## array of ids
        self.id = [0] * self.n

        ## annealing parameter
        self.rho = 0
        ## move step counter
        self.n_moves = 0
        ## target function evaluation counter
        self.n_f_evals = 0

        ## acceptance rates
        self.r_ac = []
        ## particle diversities
        self.r_pd = []

        ## The minimum distance of the marginal probability from the ; boudaries of the unit interval.
        self.eps = v['SMC_EPS']
        ## The minimum correlation required to include the component in a logistic regression.
        self.delta = v['SMC_DELTA']
        ## The efficient sample size targeted when computing the step length.
        self.eta = v['SMC_ETA']

        self.__k = numpy.array([2 ** i for i in xrange(self.d)])

        if self.verbose:
            sys.stdout.write('initializing...')
            t = time.time()

        # Check for minimum problem size.
        if self.d < v['DATA_MIN_DIM']:
            self.rho = 1.0
            self.X = list()
            self.X = numpy.array([utils.format.dec2bin(dec, self.d) for dec in xrange(2 ** self.d)])
            self.log_f = self.f.lpmf(self.X, self.job_server)
            self.log_W = self.log_f
            return
       
        self.X = self.prop.rvs(self.n, self.job_server)
        self.log_f = self.f.lpmf(self.X, self.job_server)
        self.id = numpy.dot(numpy.array(self.X, dtype=int), self.__k)
        
        if self.verbose: print '\rinitialized in %.2f sec' % (time.time() - t)

        # do first step
        self.reweight()

    def __str__(self):
        """
            @return A string containing the mean of the particle system.
        """
        return '[' + ', '.join(['%.3f' % x for x in self.get_mean()]) + ']'

    def get_csv(self):
        """
            @return A comma separated values of mean, name, evals, time, pd, ac, log_f.
        """
        return (','.join(['%.8f' % x for x in self.get_mean()]),
                ','.join([self.condition.__name__,
                          '%.3f' % (self.n_f_evals / 1000.0),
                          '%.3f' % (time.time() - self.start)]),
                ','.join(['%.5f' % x for x in self.r_pd]),
                ','.join(['%.5f' % x for x in self.r_ac]),
                ','.join(['%.5f' % x for x in self.log_f]))

    def get_mean(self):
        """
            @return Mean of the particle system.
        """
        return numpy.dot(self.get_nweights(), self.X)

    def get_ids(self, x):
        """
            @param x binary vector.
            @return Unique id.
        """
        return numpy.dot(self.__k, numpy.array(x, dtype=int))

    def get_ess(self, alpha=None):
        """ 
            Computes the effective sample size (ess).
            @param alpha advance of the geometric bridge
            @return ess
        """
        if alpha is None: w = self.log_W
        else:             w = self.log_W + alpha * self.log_f
        w = numpy.exp(w - w.max())
        w /= w.sum()
        return 1 / (pow(w, 2).sum())

    def get_particle_diversity(self):
        """
            Computes the particle diversity.
            @return particle diversity
        """
        if True:
            # pd via dictionary keys
            dic = {}
            map(operator.setitem, (dic,)*self.n, self.id, [])
            return len(dic.keys()) / float(self.n)
        else:
            # pd via numpy
            return numpy.unique(self.id, return_index=False).shape[0]


    def reweight_ess(self):
        """
            Computes an advance of the geometric bridge such that ess = tau and
            updates the log weights.q
        """

        l = 0.0; u = 1.05 - self.rho
        alpha = min(0.05, u)

        ess = self.get_ess()
        tau = self.eta * ess

        # run bi-sectional search
        for iter in xrange(30):
            if self.get_ess(alpha) < tau:
                u = alpha; alpha = 0.5 * (alpha + l)
            else:
                l = alpha; alpha = 0.5 * (alpha + u)

            if abs(l - u) < ibs.CONST_PRECISION or self.rho + l > 1.0: break

        # update rho and and log weights
        if self.rho + alpha > 1.0: alpha = 1.0 - self.rho
        utils.format.progress(ratio=self.rho + alpha, last_ratio=self.rho)
        self.rho += alpha
        self.log_W += alpha * self.log_f

        if self.verbose:
            utils.format.progress(ratio=self.rho, text='\n')
            print '\n' + str(self) + '\n'


    def slide_weigths(self):
        """
            1) shift particle components up by w_size
            2) sample uniform entries of dim w_size at the end of the particle
            3) compute weights according to the new particles
            4) fit the log cond model
            5) re-init system with log cond model 
        """
        pass


    def fit_proposal(self):
        """
            Adjust the proposal model to the particle system.
            @todo sample.distinct could be activated for speedup
        """
        if self.verbose:
            sys.stdout.write('fitting proposal...')
            t = time.time()
        sample = utils.data.data(self.X, self.log_W)
        # sample.distinct()
        self.prop.renew_from_data(sample, job_server=self.job_server, eps=self.eps, delta=self.delta, verbose=False)
        if self.verbose: print '\rfitted proposal in %.2f sec' % (time.time() - t)

    def get_nweights(self):
        """
            @return Normalized weights.
        """
        w = numpy.exp(self.log_W - max(self.log_W))
        return w / w.sum()

    def get_structure(self):
        """
            Gather a summary of how many particles are n-fold in the particle
            system.
        """
        id_set = set(self.id)
        l = [ self.id.count(i) for i in id_set ]
        k = [ l.count(i) * i for i in xrange(1, 101) ]
        return str(k) + ' %i ' % sum(k)

    def get_max(self):
        index = numpy.argmax(self.log_f)
        return self.log_f[index], self.X[index]


    #

    # Resample-Move

    #

    def resample_move(self):
        self.resample()
        self.move()

    def resample(self, augmented=False):
        """ Resamples the particle system. """

        if self.verbose:
            t = time.time()
            sys.stdout.write('resampling...')

        indices = self._resample(self.get_nweights(), numpy.random.random())

        # move objects according to resampled order
        self.id = [self.id[i] for i in indices]
        self.X = self.X[indices]
        self.log_f = self.log_f[indices]
        self.log_W = numpy.zeros(self.n)

        pD = self.pD

        # update log proposal values
        if not self.job_server is None and self.job_server.get_ncpus() > 1:
            self.log_prop = self.prop.lpmf(self.X, self.job_server)
        else:
            self.log_prop[0] = self.prop.lpmf(self.X[0])
            for i in xrange(1, self.n):
                if (self.log_prop[i] == self.log_prop[i - 1]).all():
                    self.log_prop[i] = self.log_prop[i - 1]
                else:
                    self.log_prop[i] = self.prop.lpmf(self.X[i])

        if self.verbose:
            print '\rresampled in %.2f sec, pD: %.3f' % (time.time() - t, pD)

    pD = property(fget=get_particle_diversity, doc="particle diversity")

    def move(self):
        """ 
            Moves the particle system according to an independent Metropolis-
            Hastings kernel to fight depletion of the particle system.
        """

        prev_pD = 0
        self.r_ac += [0]
        for iter in xrange(10):
            self.n_moves += 1
            accept = self.kernel()
            self.r_ac[-1] += accept
            pD = self.pD
            if self.verbose: print "moved with aR: %.3f, pD: %.3f" % (accept / float(self.n), pD)
            if pD - prev_pD < 0.04 or pD > 0.93: break
            else: prev_pD = pD

        self.r_ac[-1] /= ((iter + 1) * float(self.n))
        self.r_pd += [pD]

    def kernel(self):
        """
            Propagates the particle system via an independent Metropolis Hasting
            kernel.
        """

        self.n_f_evals += self.n

        # sample
        if self.verbose:
            sys.stdout.write('sampling...')
            t = time.time()
        Y, log_prop_Y = self.prop.rvslpmf(self.n, self.job_server)
        if self.verbose: print '\rsampled in %.2f sec' % (time.time() - t)

        # evaluate
        if self.verbose:
            sys.stdout.write('evaluating...')
            t = time.time()
        log_f_Y = self.f.lpmf(Y, self.job_server)
        if self.verbose: print '\revaluated in %.2f sec' % (time.time() - t)

        # move
        log_pi_Y = self.rho * log_f_Y
        log_pi_X = self.rho * self.log_f
        log_prop_X = self.log_prop

        accept = numpy.random.random(self.n) < numpy.exp(log_pi_Y - log_pi_X + log_prop_X - log_prop_Y)
        self.X[accept] = Y[accept]
        self.log_f[accept] = log_f_Y[accept]
        self.log_prop[accept] = log_prop_Y[accept]
        for index in xrange(self.n):
            if accept[index]:
                self.id[index] = self.get_ids(Y[index])
        return accept.sum()


    #

    # Augment-resample.

    #

    def augment_resample(self):
        """
            Augments the particle system via Metropolis-Hasting splits and
            resamples.
        """

        weights = self.get_nweights()
        ess = self.get_ess()

        while True:

            # number of weights
            m = weights.shape[0]

            # Create an index list of particles to be resampled. The number of
            # index i corresponds to the weight w_i.
            index_particles = list()
            for index, weight in enumerate(weights):
                index_particles += [index] * int(weight * m)

            # number of particles to be resampled
            l = len(index_particles)
            if l == 0: return

            # sample l proposal values Y
            Y, log_prop_Y = self.prop.rvslpmf(l, self.job_server)

            # evaluate the values log f(Y) of the proposals Y
            log_f_Y = self.f.lpmf(Y, self.job_server)
            self.n_f_evals += l

            # evaluate the log probabilities of the current target distribution
            log_pi_Y = self.log_W[index_particles] + self.rho * log_f_Y
            log_pi_X = self.log_W[index_particles] + self.rho * self.log_f[index_particles]

            # evaluate the log probabilities of the auxiliary distribution
            self.log_prop = self.prop.lpmf(self.X, self.job_server)
            log_prop_X = self.log_prop[index_particles]

            # compute the Metropolis-Hastings acceptance ratio 
            p = numpy.minimum(numpy.exp(log_pi_Y - log_pi_X + log_prop_X - log_prop_Y), numpy.ones(l))

            # multiply acceptance probabilities and weights
            weights_new = numpy.empty(l)
            for i in xrange(l):
                weights_new[i] = weights[index_particles[i]] * p[i]
                weights[index_particles[i]] *= (1 - p[i])

            # creates ids that are easy to sort
            id_Y = numpy.dot(numpy.array(Y, dtype=int), self.__k)
            id_X = numpy.dot(numpy.array(self.X, dtype=int), self.__k)

            weights = numpy.concatenate((weights, weights_new))
            self.log_W = numpy.log(weights)

            # update remaining arrays
            self.X = numpy.concatenate((self.X, Y))
            self.log_f = numpy.concatenate((self.log_f, log_f_Y))
            self.id = numpy.concatenate((id_X, id_Y))

            last_ess = ess
            ess = self.get_ess()
            if last_ess > ess or ess > self.n: break

        # Resample from augmented system.
        weights, indices = resample.resample_reductive(w=weights, u=numpy.random.random(), n=self.n,
                                                       f_select=resample.select_iterative)
        self.log_W = numpy.log(weights)

        # Update remaining arrays.
        self.X = self.X[indices]
        self.log_f = self.log_f[indices]
        self.id = self.id[indices]


    def augment_resample_unique(self):
        """
            Augments the particle system via Metropolis-Hasting splits and
            resamples.
        """

        weights = self.get_nweights()
        ess = self.get_ess()
        self.log_prop = self.prop.lpmf(self.X, self.job_server)

        while True:

            # Create an index list of particles to be resampled. The number of
            # index i corresponds to the weight w_i
            m = self.X.shape[0]
            index_particles, weight_particles = list(), list()
            for index, weight in enumerate(weights):
                if weight * m > 3:
                    index_particles += [index] * int(weight * m)
                    weight_particles += [weight / numpy.fix(weight * m)] * int(weight * m)

            # number of particles to be resampled
            l = len(index_particles)
            if l == 0: return

            # sample l proposal values Y
            Y, log_prop_Y = self.prop.rvslpmf(l, self.job_server)

            # evaluate the values log f(Y) of the proposals Y
            log_f_Y = self.f.lpmf(Y, self.job_server)
            self.n_f_evals += l

            # evaluate the log probabilities of the current target distribution
            log_pi_Y = self.log_W[index_particles] + self.rho * log_f_Y
            log_pi_X = self.log_W[index_particles] + self.rho * self.log_f[index_particles]

            # evaluate the log probabilities of the auxiliary distribution
            log_prop_X = self.log_prop[index_particles]

            # compute the Metropolis-Hastings acceptance ratio 
            p = numpy.minimum(numpy.exp(log_pi_Y - log_pi_X + log_prop_X - log_prop_Y), numpy.ones(l))

            # creates ids that are easy to sort
            id_Y = numpy.dot(numpy.array(Y, dtype=int), self.__k)

            # index set of ordered particles
            index_X = numpy.argsort(self.id)
            index_Y = numpy.argsort(id_Y)

            # multiply acceptance probabilities and weights
            weight_particles = numpy.array(weight_particles)
            weights[index_particles] = numpy.zeros(len(index_particles))
            for i in xrange(l):
                weights[index_particles[i]] += weight_particles[i] * (1 - p[i])

            j_Y = 0
            for j_X in xrange(m):
                while j_Y < l and id_Y[index_Y[j_Y]] < self.id[index_X[j_X]]:
                    i = index_Y[j_Y]
                    self.X = numpy.append(self.X, Y[i][numpy.newaxis, :], axis=0)
                    weights = numpy.append(weights, weight_particles[i] * p[i])
                    self.log_prop = numpy.append(self.log_prop, log_prop_Y[i])
                    self.log_f = numpy.append(self.log_f, log_f_Y[i])
                    self.id = numpy.append(self.id, id_Y[i])
                    j_Y += 1
                # transfer weight from w_Y to w_X
                while j_Y < l and id_Y[index_Y[j_Y]] == self.id[index_X[j_X]]:
                    i = index_Y[j_Y]
                    weights[index_X[j_X]] += weight_particles[i] * p[i]
                    j_Y += 1

            if j_Y < l:
                i = index_Y[j_Y]
                self.X = numpy.append(self.X, Y[i][numpy.newaxis, :], axis=0)
                weights = numpy.append(weights, weight_particles[i] * p[i])
                self.log_prop = numpy.append(self.log_prop, log_prop_Y[i])
                self.log_f = numpy.append(self.log_f, log_f_Y[i])
                self.id = numpy.append(self.id, id_Y[i])
                j_Y += 1

            self.log_W = numpy.log(weights)
            last_ess = ess
            ess = self.get_ess()
            if last_ess > ess or ess > self.n: break

        # Resample from the augmented unique system.
        weights, indices = resample.resample_reductive(w=weights, u=numpy.random.random(), n=self.n,
                                                       f_select=resample.select_iterative)
        self.log_W = numpy.log(weights)

        # Update remaining arrays.
        self.X = self.X[indices]
        self.log_prop = self.log_prop[indices]
        self.log_f = self.log_f[indices]
        self.id = self.id[indices]


def main():
    pass

if __name__ == "__main__":
    main()
