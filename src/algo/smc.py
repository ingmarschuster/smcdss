#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Sequential Monte Carlo on binary spaces.
    @namespace ibs.smc
"""

from ps import ParticleSystem
import datetime
import sys
import time

FULL_IS = 1
LAPLACE_PLUS_IS = 2

class AnnealedSMC():

    def __init__(self, param, target=1.0, job_server=None, gui=None):
        self.ps = ParticleSystem(param, job_server)
        self.target = target
        self.gui = gui
        self.param = param

    def sample(self, lock=None):

        time.sleep(0.2)
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

        if self.ps.f.criterion == LAPLACE_PLUS_IS:
            t = time.time()
            if self.ps.verbose:
                sys.stdout.write('reweight particles according to IS estimate...')
            X = self.ps.get_distinct()[0]
            self.ps.f.criterion = FULL_IS
            self.ps.log_weights = self.ps.f.lpmf(X)
            self.ps.X = X
            if self.ps.verbose:
                sys.stdout.write('\rreweighted in %.2f sec\n' % (time.time() - t))
            self.check_gui()
            
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
                ','.join(['%.3f' % (self.ps.n_f_evals / 1000.0),
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
                self.gui.mygraph.values = [self.ps.r_ac, self.ps.r_pd]
                self.gui.mygraph.lines = [self.ps.r_rs]
                self.gui.mygraph.redraw()
            if not self.gui.mybarplot is None:
                self.gui.mybarplot.values = self.ps.get_mean()
                self.gui.mybarplot.redraw()
            self.gui.progress_bar.set_value(self.ps.rho / self.target)
        else:
            self.gui.write('\rstopped.')
            self.gui.stop_button.config(relief='raised', state='normal')

        # check if running
        return self.gui.is_running
