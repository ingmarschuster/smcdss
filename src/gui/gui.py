#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    GUI.
    @namespace ibs.ibs_gui
    @details GUI.
"""

from algo.smc import AnnealedSMC
import Tkinter as tk
import config
import numpy
import os
import parallel.pp as pp
import plot
import shutil
import sys
import time
import tkMessageBox
import utils.pmw.Pmw as pmw

BUTTON = {'width':8}
STANDARD = {'padx':5, 'pady':5}

HEIGTH = 400
WIDTH = 800
PADDING = 100

GREEK_RHO = u'\u03C1'

class App(tk.Tk):

    def __init__(self):

        tk.Tk.__init__(self)

        self.title('SMC on binary spaces')
        self.smc = None
        self.job_server = None

        self.default_filename = config.get_default_filename()
        self.default = config.import_config(self.default_filename)

        self.myconfig_filename = self.default_filename
        self.myconfig = config.import_config(self.default_filename)

        self.config_window = None

        # parameter group
        group = pmw.Group(self, tag_text='Parameter')
        group.pack(expand=1, fill='both', **STANDARD)
        group = group.interior()
        fields = list()
        for name in ['status', 'target']:
            fields += [pmw.EntryField(group,
                         entry_width=8, labelpos='w', value='0.0',
                         label_text=name + ' ' + GREEK_RHO,
                         validate={'validator' : 'real', 'min' : 0.0},
                         command=self.focus_set)]
            fields[-1].pack(side=tk.LEFT, anchor=tk.NE, **STANDARD)
        self.rho, self.target = tuple(fields)

        # startup group
        group = pmw.Group(self, tag_text='Dashboard')
        group.pack(expand=1, fill='both', **STANDARD)
        group = group.interior()
        self.dashboard = {}
        for label in ['data set']:
            self.dashboard[label] = pmw.EntryField(group, entry_width=16, labelpos='w', label_text=label)
            self.dashboard[label].pack(anchor=tk.NW, expand=1, fill='x', **STANDARD)

        for text, command in [('Config', self.open_config_window),
                              ('Start', self.start),
                              ('Stop', self.stop)]:
            tk.Button(group, text=text, command=command,
                      **BUTTON).pack(side=tk.RIGHT, anchor=tk.SE, **STANDARD)

        # configuration group
        group = pmw.Group(self, tag_text='Configuration')
        group.pack(expand=1, fill='both', **STANDARD)
        group = group.interior()
        self.box_run_file = pmw.ScrolledListBox(group, labelpos='nw', label_text='File',
                                                selectioncommand=self.load_run_file,
                                                dblclickcommand=self.open_config_window)
        lb = self.box_run_file._listbox
        lb.bind('<Up>', lambda event : self.on_arrow_keys(-1))
        lb.bind('<Down>', lambda event : self.on_arrow_keys(1))
        lb.focus_force()
        self.box_run_file.pack(fill='both', **STANDARD)

        for text, command in [('Refresh', self.refresh_run_file),
                              ('Copy', lambda : self.copy_run_file()),
                              ('Delete', self.delete_run_file)]:
            tk.Button(group, text=text, command=command,
                      **BUTTON).pack(side=tk.RIGHT, anchor=tk.SE, **STANDARD)

        self.refresh_run_file()

        self.target.setvalue(1.0)

    def on_arrow_keys(self, move):
        '''
            Update list box.
        '''
        lb = self.box_run_file._listbox
        lb.select_clear(lb.index(tk.ACTIVE))
        index = min(max(lb.index(tk.ACTIVE) + move, 0), lb.size() - 1)
        lb.select_set(index)
        self.load_run_file()

    def copy_run_file(self):
        FileCopyDialog(self)

    def delete_run_file(self):
        '''
            Delete selected run file.
        '''
        if self.myconfig_filename == self.default_filename: return
        if tkMessageBox.askyesno("Delete", "Delete the configuration '%s'?" % os.path.basename(self.myconfig_filename)):
            os.remove(self.myconfig_filename)
            self.refresh_run_file()

    def load_run_file(self, event=None):
        '''
            Select a run file.
        '''
        self.myconfig_filename = self.box_run_file.getvalue()[0]
        if self.myconfig_filename == 'default':
            self.myconfig_filename = self.default_filename
        else:
            self.myconfig_filename = os.path.join(self.default['path/run'], self.myconfig_filename)
        self.myconfig = config.import_config(self.myconfig_filename)

        # update dashboard
        self.dashboard['data set'].setvalue(os.path.basename(self.myconfig['data/csv_file']))

        # update configuration window
        if not self.config_window is None:
            self.config_window.filename = self.myconfig_filename
            self.config_window.reset()

    def refresh_run_file(self):
        '''
            Refresh the run file box.
        '''
        # remember current selection
        try: item = self.box_run_file.getvalue()[0]
        except IndexError: item = 'default'

        files = ['default'] + [filename for filename in os.listdir(self.default['path/run'])
                              if filename[-4:] == '.ini']
        if not item in files: item = 'default'

        # refresh box
        self.box_run_file.setlist(files)
        self.box_run_file._listbox.activate(files.index(item))
        self.box_run_file.setvalue(item)

        # reload configuration
        self.myconfig_filename = self.box_run_file.getvalue()[0]
        self.load_run_file(self.myconfig_filename)

    def open_config_window(self):
        '''
            Open configuration window.
        '''
        if self.config_window is None:
            self.config_window = config.GuiConfig(self.master, self.myconfig_filename)

    def stop(self):
        '''
            Stop running algorithm.
        '''
        self.is_running = False

    def start(self):
        '''
            Start algorithm.
        '''
        config.import_data(self.myconfig)
        self.is_running = True

        # initialize job server
        if self.job_server is None and not self.myconfig['run/cpus'] is None:
            sys.stdout.write('Starting jobserver...')
            t = time.time()
            self.myconfig['run/job_server'] = pp.Server(ncpus=self.myconfig['run/cpus'], ppservers=())
            print '\rJob server (%i) started in %.2f sec' % (self.myconfig['run/job_server'].get_ncpus(), time.time() - t)

        # initialize SMC
        if self.smc is None or self.smc.ps.rho >= float(self.target.getvalue()):
            self.mygraph, self.mybarplot = None, None
            target = float(self.target.getvalue())
            self.smc = AnnealedSMC(param=self.myconfig, target=target, job_server=self.job_server, gui=self)
        else:
            self.smc.target = float(self.target.getvalue())

        # initialize plots
        if self.mygraph is None:
            t = self.winfo_toplevel()
            self.mygraph = plot.GuiGraph(self,
                                  x=t.winfo_x() + t.winfo_width() + 10,
                                  y=50,
                                  w=WIDTH, h=HEIGTH,
                                  legend=['accceptance rate', 'particle diversity'])
        if self.mybarplot is None:
            t = self.winfo_toplevel()
            self.mybarplot = plot.GuiBarPlot(self,
                                  x=t.winfo_x() + t.winfo_width() + 10,
                                  y=HEIGTH + PADDING,
                                  w=WIDTH, h=HEIGTH,
                                  labels=numpy.arange(self.myconfig['f'].d) + 1)
            self.mybarplot.values = self.smc.ps.get_mean()
            self.mybarplot.redraw()

        # run SMC
        self.smc.sample()


class FileCopyDialog(tk.Toplevel):

    def __init__(self, master):
        '''
            Constructor.
        '''
        self.master = master
        tk.Toplevel.__init__(self, master)
        self.title('Enter file name...')
        self.field = pmw.EntryField(self, labelpos='w', entry_width=24,
                                    value='copy_of_' + os.path.basename(self.master.myconfig_filename),
                                    label_text='New file name: ')
        self.field.pack(**STANDARD)
        tk.Button(self, text="Okay", command=self.okay).pack(side='right', **STANDARD)
        tk.Button(self, text="Cancel", command=self.close).pack(side='right', **STANDARD)
        self.grab_set()
        self.center_window()

    def close(self):
        '''
            Close dialog.
        '''
        self.destroy()

    def okay(self):
        '''
            Copy file.
        '''
        src = self.master.myconfig_filename
        dst = os.path.join(os.path.dirname(src), self.field.getvalue())
        if self.field.getvalue() == '' or  src == dst: return
        shutil.copy(src, dst)
        self.master.refresh_run_file()
        self.close()

    def center_window(self):
        '''
            Center the configuration window.
        '''
        w, h = 300, 75
        ws = self.winfo_screenwidth()
        hs = self.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        self.geometry('%dx%d+%d+%d' % (w, h, x, y))

if __name__ == '__main__':
    App().mainloop()
