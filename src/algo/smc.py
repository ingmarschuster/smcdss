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

class AnnealedSMC():

    def __init__(self, param, target=1.0, job_server=None, gui=None):

        self.ps = ParticleSystem(param, job_server)
        self.target = target
        self.gui = gui

        if self.ps.d < param['data/min_dim']:
            self.ps.enumerate_state_space(target)
            self.ps.rho = target
        else:
            self.ps.initialize(param, target)

    def sample(self):

        self.start = time.time()

        # run sequential MC scheme
        while self.ps.rho < self.target:

            self.ps.fit_proposal()
            self.ps.resample()
            self.ps.move()
            self.ps.reweight(self.target)
            if not self.check_gui(): return

        sys.stdout.write('\rannealed smc completed in %s.\n' % self.get_time_elapsed())

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
                self.gui.mygraph.values = [self.ps.r_ac, self.ps.r_pd]
                self.gui.mygraph.lines = [self.ps.r_rs]
                self.gui.mygraph.redraw()
            if not self.gui.mybarplot is None:
                self.gui.mybarplot.values = self.ps.get_mean()
                self.gui.mybarplot.redraw()
            self.gui.progress_bar.set_value(self.ps.rho / self.target)

        # check if running
        return self.gui.is_running
