#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    GUI.
    @namespace gui.gui
    @details GUI.
"""

from algo.smc import AnnealedSMC
from algo.mcmc import MCMC
import Tkinter as tk
import config
import numpy
import os
import plot
import shell
import shutil
import subprocess
import sys
import threading
import datetime
import time
import tkColorChooser as tkcc
import tkMessageBox
import utils.logger
import utils.pmw.Pmw as pmw
import utils.pmw.meter as meter

BUTTON = {'width':6}
STANDARD = {'padx':5, 'pady':5}
THREAD = True
VERBOSE = True
HEIGTH = 200
WIDTH = 600
PADDING = 100

GREEK_RHO = u'\u03C1'

class App(tk.Tk):

    def __init__(self):

        tk.Tk.__init__(self)

        self.myalgo = None
        self.myalgo_thread = None
        self.mygraph = None
        self.mybarplot = None
        self.myplots = list()
        self.mytime = None

        self.config_window = None
        self.is_running = False
        self.job_server = None

        self.default_filename = config.get_default_filename()
        self.default = config.import_config(self.default_filename)

        self.myconfig_filename = self.default_filename
        self.myconfig = config.import_config(self.default_filename)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=3)

        self.bar_color = '#33ffff'
        if VERBOSE:
            sys.stdout = utils.logger.Logger(sys.stdout, textfield=self)
        else:
            sys.stdout = utils.logger.Logger(textfield=self)

        # dashboard group
        #
        group = pmw.Group(self, tag_text='Dashboard')
        group.grid(row=0, column=0, sticky='nswe', **STANDARD)
        group = group.interior()
        group.rowconfigure(1, weight=1)
        group.columnconfigure(0, weight=1)

        self.progress_bar = meter.Meter(group, width=250, fillcolor=self.bar_color, relief='ridge', bd=2)
        self.progress_bar.\
            grid(row=0, column=0, sticky='ew', **STANDARD)
        tk.Label(group, text='target ' + GREEK_RHO). \
            grid(row=0, column=1, sticky='w', **STANDARD)
        self.target = pmw.EntryField(group, entry_width=8, value='0.0',
                                     validate={'validator' : 'real', 'min' : 0.0})
        self.target.\
            grid(row=0, column=2, sticky='ew', **STANDARD)

        self.progress_news = tk.Text(group, width=4, height=4)
        self.progress_news. \
            grid(row=1, column=0, columnspan=3, sticky='nsew', **STANDARD)

        frame = tk.Frame(group)
        buttons = {}
        frame.grid(row=2, column=0, columnspan=3, sticky='wse', **STANDARD)
        for text, command in [('Start', self.start),
                              ('Stop', self.stop),
                              ('Reset', self.reset)]:
            buttons[text] = tk.Button(frame, text=text, command=command, **BUTTON)
            buttons[text].pack(side='right', **STANDARD)
        self.stop_button = buttons['Stop']

        # clock
        self.curtime = ''
        self.clock = tk.Label(frame, borderwidth=1, width=8, relief='sunken', bg='white', padx=10, pady=4)
        self.clock.pack(side='left', **STANDARD)
        self.tick()

        frame = tk.Frame(group)
        frame.grid(row=3, column=0, columnspan=3, sticky='ew', **STANDARD)
        tk.Button(frame, text='External', command=self.start_external, **BUTTON).pack(side='right', **STANDARD)
        self.external_processes = tk.Spinbox(frame, values=range(1, 512), width=4)
        self.external_processes.pack(side='right', **STANDARD)
        tk.Button(frame, text='PDF', command=lambda : shell.show_pdf(self.myconfig), **BUTTON).pack(side='left', **STANDARD)
        self.write_file = tk.IntVar()
        tk.Checkbutton(frame, text='write file', variable=self.write_file, width=8).pack(side='left', **STANDARD)
        self.write_clean = tk.IntVar()
        tk.Checkbutton(frame, text='write clean', variable=self.write_clean, width=8).pack(side='left', **STANDARD)

        # configuration group
        #
        group = pmw.Group(self, tag_text='Configuration')
        group.grid(row=1, column=0, sticky='nswe', **STANDARD)
        group = group.interior()
        group.rowconfigure(0, weight=1)
        group.columnconfigure(0, weight=1)
        self.box_run_file = pmw.ScrolledListBox(group, labelpos='nw', label_text='File',
                                                selectioncommand=self.load_run_file,
                                                dblclickcommand=self.open_config_window,
                                                listbox_height=15)
        lb = self.box_run_file._listbox
        lb.bind('<F2>', lambda event : self.rename_run_file())
        lb.bind('<Up>', lambda event : self.on_arrow_keys(-1))
        lb.bind('<Down>', lambda event : self.on_arrow_keys(1))
        lb.focus_force()
        self.box_run_file.grid(row=0, column=0, columnspan=3, sticky='nswe', **STANDARD)

        self.data_set = pmw.EntryField(group, entry_width=16,
                                     labelpos='w', label_text='data set')
        self.data_set. \
            grid(row=1, column=0, sticky='ew', **STANDARD)
        tk.Label(group, text='color'). \
            grid(row=1, column=1, sticky='w', **STANDARD)
        self.bar_color_button = tk.Button(group, command=self.set_bar_color, width=4,
                                          activebackground=self.bar_color, background=self.bar_color)
        self.bar_color_button. \
            grid(row=1, column=2, sticky='ew', **STANDARD)

        button_frame = tk.Frame(group)
        buttons = {}
        button_frame.grid(row=2, column=0, columnspan=3, sticky='se', **STANDARD)
        for text, command in [('Refresh', self.refresh_run_file),
                              ('Copy', lambda : self.copy_run_file()),
                              ('Rename', lambda : self.rename_run_file()),
                              ('Delete', self.delete_run_file)]:
            buttons[text] = tk.Button(button_frame, text=text, command=command, **BUTTON)
            buttons[text].pack(side='right', **STANDARD)

        self.refresh_run_file()

        self.target.setvalue(1.0)

        self.protocol("WM_DELETE_WINDOW", self.close_window)

    def tick(self):
        if self.mytime is None: return
        newtime = str(datetime.timedelta(seconds=time.time() - self.mytime))[:9]
        if newtime != self.curtime:
            self.curtime = newtime
            self.clock.config(text=self.curtime)
            self.clock.after(100, self.tick)

    def start_timer(self):
        self.mytime = time.time()
        self.tick()

    def stop_timer(self):
        self.mytime = None

    def close_window(self):
        self.reset(kill_plots=True)
        self.destroy()

    def on_arrow_keys(self, move):
        '''
            Update list box.
        '''
        lb = self.box_run_file._listbox
        lb.select_clear(lb.index(tk.ACTIVE))
        index = min(max(lb.index(tk.ACTIVE) + move, 0), lb.size() - 1)
        lb.select_set(index)
        self.load_run_file()

    def set_bar_color(self, bar_color=None):
        '''
            Set color.
        '''
        if bar_color is None:
            bar_color = tkcc.askcolor(initialcolor=self.bar_color)[1]
        if bar_color is None: return
        self.bar_color = bar_color
        self.bar_color_button.config(background=self.bar_color, activebackground=bar_color)
        self.progress_bar.set_color(bar_color)
        self.myconfig['layout/color'] = bar_color

        # save color to INI file
        temp = config.read_config(self.myconfig_filename)
        temp['eval']['layout']['color'] = bar_color
        config.write_config(temp)

        if not self.mybarplot is None:
            self.mybarplot.bar_color = bar_color
            self.mybarplot.redraw()

    def write(self, text):
        ''' Write to text field. '''
        self.progress_news.config(state=tk.NORMAL)
        if '\r' in text:
            text = text.replace('\r', '')
            self.progress_news.delete(1.0, tk.END)
        self.progress_news.insert(tk.END, text)
        self.progress_news.update()
        self.progress_news.config(state=tk.DISABLED)

    def write_result_file(self):
        ''' Write result file. '''
        if self.write_file.get():
            shell.write_result_file(self.myalgo.get_csv(), 0, self.myconfig)

    def copy_run_file(self):
        ''' Open file copy dialog. '''
        FileDialog(self, mode='copy')

    def rename_run_file(self):
        ''' Open file copy dialog. '''
        FileDialog(self, mode='rename')

    def delete_run_file(self):
        ''' Delete selected run file. '''
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
        if not self.is_running: self.reset(kill_plots=False)
        self.data_set.setvalue(os.path.basename(self.myconfig['data/csv_file']))
        self.title("Sampling '%s'" % self.myconfig['run/name'])

        # update configuration window
        if not self.config_window is None:
            self.config_window.filename = self.myconfig_filename
            self.config_window.reset()

        self.set_bar_color(self.myconfig['layout/color'])

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
        self.stop_timer()
        if self.is_running:
            self.write('\rstopping...')
            self.is_running = False
            self.stop_button.config(disabledforeground='#dd0000', relief='sunken', state='disabled')
            self.box_run_file._listbox.config(state='normal')
            if not self.job_server is None:
                self.job_server.destroy()
                self.job_server = None

    def reset(self, kill_plots=True):
        '''
            Stop running algorithm.
        '''
        if self.is_running:
            self.stop()
            return
        self.progress_bar.set_value(0.0)
        self.myalgo = None
        self.job_server = None
        self.write('\r')

        if kill_plots:
            for plot in self.myplots:
                try:
                    plot.destroy()
                except:
                    pass
            self.mybarplot = None
            self.mygraph = None

    def start_external(self):
        '''
            Start algorithm as external process.
        '''
        if self.myconfig_filename == config.get_default_filename(): return
        if self.write_clean.get():
            shell.clean_folder(self.myconfig)
        k = int(self.external_processes.get())
        f = os.path.basename(self.myconfig_filename)
        self.write('\rStart %d external instances of %s.' % (k, f))
        while k > 0:
            if os.name == 'posix':
                subprocess.Popen(['gnome-terminal -e "ibs -r %s"' % f], shell=True)
            else:
                path = os.path.abspath(os.path.join(os.path.join(*([os.getcwd()] + ['..'] * 1)), 'bin', 'ibs.bat'))
                subprocess.call('start "ibs" /MAX "%s" -r %s' % (path, f), shell=True)
            k -= 1

    def start(self):
        '''
            Start algorithm.
        '''
        self.start_timer()
        self.myconfig = config.import_data(self.myconfig)
        self.myconfig['run/verbose'] = True
        self.is_running = True
        self.box_run_file._listbox.config(state='disabled')

        # initialize job server
        if self.job_server is None:
            self.job_server = shell.prepare_job_server(self.myconfig)

        # initialize algorithm
        if self.myalgo is None or self.myalgo.ps.rho >= float(self.target.getvalue()):
            if self.write_clean.get():
                shell.clean_folder(self.myconfig)
            if self.write_file.get():
                shell.prepare_run(self.myconfig)
            target = float(self.target.getvalue())

            if self.myconfig['run/algo'].lower() == 'smc':
                self.myalgo = AnnealedSMC(param=self.myconfig, target=target, job_server=self.job_server, gui=self)
            if self.myconfig['run/algo'].lower() == 'mcmc':
                self.myalgo = MCMC(param=self.myconfig, job_server=self.job_server, gui=self)
            self.prepare_plots(values=0.5 * numpy.ones(self.myalgo.ps.d))

            # init
            if THREAD:
                lock = threading.Lock()
                t = threading.Thread(target=self.myalgo.initialize, args=(lock,))
                t.daemon = True
                t.start()
            else:
                self.myalgo.initialize()
        else:
            lock = threading.Lock()
            self.myalgo.target = float(self.target.getvalue())

        # run
        if THREAD:
            t = threading.Thread(target=self.myalgo.sample, args=(lock,))
            t.daemon = True
            t.start()
        else:
            self.myalgo.sample()

    def prepare_plots(self, values=None):
        '''
            Open plots.
        '''
        t = self.winfo_toplevel()
        algo = self.myconfig['run/algo'].lower()
        self.mygraph = plot.GuiGraph(self,
                              x=t.winfo_x() + t.winfo_width() + 10,
                              y=50,
                              w=WIDTH, h=HEIGTH,
                              legend={'smc':['accceptance rate', 'particle diversity', 'resampling'],
                                      'mcmc':['accceptance rate', 'bits flipped']}[algo],
                              adjust_scale={'smc':False, 'mcmc':True}[algo],
                              first_val={'smc':1.0, 'mcmc':0.0}[algo])

        t = self.winfo_toplevel()

        self.mybarplot = plot.GuiBarPlot(self,
                              x=t.winfo_x() + t.winfo_width() + 10,
                              y=HEIGTH + PADDING,
                              w=WIDTH, h=2 * HEIGTH,
                              bar_color=self.bar_color,
                              labels=self.myconfig['data/free_header'])
        if values is None: values = self.myalgo.ps.get_mean()
        self.mybarplot.values = values
        self.mybarplot.redraw()
        self.myplots += [self.mygraph, self.mybarplot]


class FileDialog(tk.Toplevel):

    def __init__(self, master, mode='copy'):
        '''
            Constructor.
        '''
        value = os.path.basename(master.myconfig_filename)
        self.mode = mode
        if self.mode == 'copy': value = 'copy_of_' + value

        self.master = master
        tk.Toplevel.__init__(self, master)
        self.title('Enter file name...')
        self.field = pmw.EntryField(self, labelpos='w', entry_width=24,
                                    value=value,
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
        if self.mode == 'copy':
            shutil.copy(src, dst)
        if self.mode == 'rename':
            shutil.move(src, dst)
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
