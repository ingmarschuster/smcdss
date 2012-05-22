#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Markov chain Monte Carlo on binary spaces.
    @namespace ibs.mcmc
"""

import time
import sys
import datetime
import mc

BURNIN_COLOR = "#999999"

class MCMC():

    def __init__(self, param, gui=None, job_server=None):
        self.ps = mc.MarkovChain(param)
        self.gui = gui
        self.param = param

    def initialize(self, lock=None):
        """
            Initialize Markov chain.
        """
        if not lock is None:
            lock.acquire()
        if self.ps.d < self.param['data/min_dim']:
            self.ps.enumerate_state_space(self.target)
            return

        t = time.time()
        self.ps.max_evals /= 100.0
        self.ps.chunk_size = self.ps.max_evals / 20.0
        sys.stdout.write('%s: %i steps burn in...' % (self.ps.kernel.name.lower(), self.ps.max_evals))

        if not self.gui is None:
            bar_color = self.gui.bar_color
            self.gui.progress_bar.set_color(BURNIN_COLOR)
            if not self.gui.mybarplot is None: self.gui.mybarplot.bar_color = BURNIN_COLOR

        while not self.ps.done:
            self.ps.do_step()
            if not self.check_gui(): return

        sys.stdout.write('\rburn in completed in %.2f sec\n' % (time.time() - t))
        self.ps = mc.MarkovChain(self.param, x=self.ps.x)

        if not self.gui is None:
            self.gui.progress_bar.set_color(bar_color)
            if not self.gui.mybarplot is None: self.gui.mybarplot.bar_color = bar_color
            self.check_gui()

        if not lock is None:
            lock.release()

    def sample(self, lock=None):
        """
            Compute an estimate of the expected value via MCMC.
            \param v parameters
            \param verbose verbose
        """

        if not lock is None:
            lock.acquire()

        time.sleep(0.2)
        self.start = time.time()

        # run for maximum time or maximum iterations
        while not self.ps.done:
            self.ps.do_step()
            if not self.check_gui(): return

        sys.stdout.write('\rmarkov chain monte carlo completed in %s.\n' % self.get_time_elapsed())

        # finalize thread
        if not self.gui is None:
            self.gui.stop()
            self.gui.write_result_file()
            self.check_gui()

    def get_time_elapsed(self):
        """
            \return Get elapsed time.
        """
        return str(datetime.timedelta(seconds=time.time() - self.start))

    def get_csv(self):
        """
            \return A comma separated values of mean, evals, length, moves, time
        """
        return (','.join(['%.8f' % x for x in self.ps.get_mean()]),
                ','.join(['%.3f' % (self.ps.n_f_evals * 1e-3),
                          '%.3f' % (self.ps.length * 1e-3),
                          '%.3f' % (self.ps.n_moves * 1e-3),
                          '%.3f' % (time.time() - self.start)]),
                ','.join(['%.5f' % x for x in self.ps.r_ac]))

    def check_gui(self):
        """
            Plots advance of SMC on GUI.
        """
        if self.gui is None: return True
        # show status
        if self.gui.is_running:
            if not self.gui.mygraph is None:
                self.gui.mygraph.values = [self.ps.r_ac, self.ps.r_bf]
                self.gui.mygraph.redraw()
            if not self.gui.mybarplot is None:
                self.gui.mybarplot.values = self.ps.get_mean()
                self.gui.mybarplot.redraw()
            self.gui.progress_bar.set_value(self.ps.rho)
        else:
            self.gui.write('\rstopped.')
            self.gui.stop_button.config(relief='raised', state='normal')

        # check if running
        return self.gui.is_running
