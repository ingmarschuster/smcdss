#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Sequential Monte Carlo on binary spaces.
    @namespace ibs.smc
"""

from ps import ParticleSystem
from binary.selector_glm import link
import datetime
import numpy
import os
import sys
import time

class AnnealedSMC():

    def __init__(self, param, target=1.0, job_server=None, gui=None):

        self.ps = ParticleSystem(param, job_server)
        self.target = target
        self.gui = gui
        self.param = param

    def sample(self, lock=None):
        if not lock is None:
            lock.acquire()
        self.start = time.time()

        # run sequential MC scheme
        while self.ps.rho < self.target:

            self.ps.fit_proposal()
            self.ps.resample()
            self.ps.move()
            self.ps.reweight(self.target)
            if not self.check_gui(): return

        sys.stdout.write('\rannealed smc completed in %s.\n' % self.get_time_elapsed())
        if not lock is None:
            lock.release()

        # finalize thread
        if not self.gui is None:
            self.gui.stop()
            self.gui.write_result_file()
            self.check_gui()

    def initialize(self, lock=None):
        """
            Initialize particle system.
        """
        if not lock is None:
            lock.acquire()
        if self.ps.d < self.param['data/min_dim']:
            self.ps.enumerate_state_space(self.target)
        else:
            self.ps.initialize(self.param, self.target)
        if not lock is None:
            lock.release()
        self.check_gui()

    def get_time_elapsed(self):
        """
            \return Get elapsed time.
        """
        return str(datetime.timedelta(seconds=time.time() - self.start))

    def get_csv(self):
        """
            \return A comma separated values of mean, name, evals, time, pd, ac, log_f.
        """
        return (','.join(['%.8f' % x for x in self.ps.get_mean()]),
                ','.join(['resample-move',
                          '%.3f' % (self.ps.n_f_evals / 1000.0),
                          '%.3f' % (time.time() - self.start)]),
                ','.join(['%.5f' % x for x in self.ps.r_pd]),
                ','.join(['%.5f' % x for x in self.ps.r_ac]))

    def check_gui(self):
        """
            Plots advance of SMC on GUI.
        """
        if self.gui is None: return True
        # show status
        if self.gui.is_running:
            if not self.gui.mygraph is None:
                self.gui.mygraph.lock.acquire()
                self.gui.mygraph.values = [self.ps.r_ac, self.ps.r_pd]
                self.gui.mygraph.lines = [self.ps.r_rs]
                self.gui.mygraph.redraw()
                self.gui.mygraph.lock.release()
            if not self.gui.mybarplot is None:
                print 'smc locks'
                self.gui.mygraph.lock.acquire()
                print 'smc after locks'
                self.gui.mybarplot.values = self.ps.get_mean()
                self.gui.mybarplot.redraw()
                self.gui.mygraph.lock.release()
            self.gui.progress_bar.set_value(self.ps.rho / self.target)
        else:
            self.gui.write('\rstopped.')
            self.gui.stop_button.config(relief='raised', state='normal')

        # check if running
        return self.gui.is_running


def main():

    path = os.path.expanduser('~/Documents/Data/bvs/test')
    n = 300
    p = 8
    d = 20

    #beta = numpy.random.standard_normal(size=p)
    beta = numpy.array([-2, -2, -1, -1, 1, 1, 2, 2])
    beta_0 = numpy.random.randn()

    X = numpy.random.standard_normal(size=(n, d))
    y = numpy.zeros(n, dtype=int)
    for k in xrange(n):
        prob = link(beta_0 + numpy.dot(X[k, :p], beta))
        if numpy.random.random() < prob:
            y[k] = 1

    f = open(os.path.join(path, 'test_link.csv'), 'w')
    f.write(','.join(['y'] + ['x%d' % (i + 1) for i in xrange(d)]) + '\n')
    for k in xrange(n):
        f.write(','.join(['%d' % y[k]] + ['%.6f' % x for x in X[k]]) + '\n')
    f.close()

if __name__ == "__main__":
    main()
