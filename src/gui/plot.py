# -*- coding: utf-8 -*-

"""
GUI plots.
"""

"""
@namespace ibs.ibs_plot
@details GUI.
"""

import Tkinter as tk
import csv
import datetime
import numpy
import os
import subprocess
import sys
import utils.pmw.Pmw as pmw

TOP_BORDER = 10
BOTTOM_BORDER = 10
LEFT_BORDER = 40
RIGHT_BORDER = 30
BARPLOT_COLORS = ['#33ffff']
GRAPH_PLOTCHARS = ['x', 'o']
GRAPH_COLORS = ['#990000', '#009900']

GREEK_RHO = u'\u03C1'


class GuiPlot(tk.Toplevel):

    def __init__(self, master, x=10, y=10, w=400, h=400, labels=None, legend=None):

        tk.Toplevel.__init__(self, master, height=h, width=w)
        self.geometry('+%d+%d' % (x, y))

        self.legend = legend
        self.labels = labels

        self.values = None
        self.pressed_ctrl = False
        self.pressed_alt = False
        self.wheel_event = False

        # setup scrolled canvas
        self.sc = pmw.ScrolledCanvas(self, borderframe=1)
        self.c = self.sc._canvas
        self.c.configure(width=w, height=h, background='white')
        self.sc.pack(fill='both', expand=1)

        # initialize empty plot
        self.plot_height = h - TOP_BORDER - BOTTOM_BORDER
        self.plot_width = w - LEFT_BORDER - RIGHT_BORDER
        self.draw_axis()
        self.sc.resizescrollregion()

        # setup bindings
        self.c.bind('<Configure>', self.resize_window_handler)
        self.c.bind('<4>', lambda event : self.wheel_handler(-1))
        self.c.bind('<5>', lambda event : self.wheel_handler(1))
        self.bind("<KeyPress-Control_L>", lambda event : self.set_pressed_ctrl(True))
        self.bind("<KeyRelease-Control_L>", lambda event : self.set_pressed_ctrl(False))
        self.bind("<KeyPress-Alt_L>", lambda event : self.set_pressed_alt(True))
        self.bind("<KeyRelease-Alt_L>", lambda event : self.set_pressed_alt(False))

        self.protocol("WM_DELETE_WINDOW", self.close_window)

    def set_pressed_ctrl(self, mode):
        self.pressed_ctrl = mode

    def set_pressed_alt(self, mode):
        self.pressed_alt = mode

    def wheel_handler(self, change):
        self.wheel_event = True
        if self.pressed_ctrl:
            self.plot_width -= change * 25
            self.draw_axis()
            self.redraw()
            self.sc.resizescrollregion()
        elif self.pressed_alt:
            self.c.xview_scroll(change, 'units')
        else:
            self.c.yview_scroll(change, 'units')

    def resize_window_handler(self, event):
        if self.pressed_ctrl and not self.wheel_event:
            self.plot_height = event.height - TOP_BORDER - BOTTOM_BORDER
            self.plot_width = event.width - LEFT_BORDER - RIGHT_BORDER
            self.draw_axis()
            self.redraw()
            self.sc.resizescrollregion()
        self.wheel_event = False

    def redraw(self):
        self.c.update()

    def draw_axis(self):
        # remove old axis
        self.c.delete('axis')

        # add axis
        self.c.create_line(LEFT_BORDER - 10,
                      TOP_BORDER,
                      LEFT_BORDER - 10,
                      TOP_BORDER + self.plot_height, tag='axis')
        for i, y in enumerate(numpy.linspace(0.0, self.plot_height, 11)):
            self.c.create_line(LEFT_BORDER - 10,
                          TOP_BORDER + y,
                          LEFT_BORDER - 5,
                          TOP_BORDER + y, tag='axis')
            self.c.create_line(LEFT_BORDER,
                          TOP_BORDER + y,
                          LEFT_BORDER + self.plot_width,
                          TOP_BORDER + y, stipple="gray25", fill="#000077", tag='axis')
            self.c.create_text(3, TOP_BORDER + y, text=str(1 - i / 10.0), anchor=tk.W, tag='axis')

        # add labels
        if not self.labels is None:
            n = self.labels.shape[0]
            box_width = self.plot_width / float(n)
            for i, x in enumerate(numpy.linspace(box_width, self.plot_width, n)):
                self.c.create_text(LEFT_BORDER + x - box_width / 2.0,
                              TOP_BORDER + self.plot_height + BOTTOM_BORDER,
                              text=self.labels[i], tag='axis', width=3, anchor=tk.N)

        # add legend
        if not self.legend is None:
            n = len(self.legend)
            self.c.create_rectangle(LEFT_BORDER + 5,
                                    TOP_BORDER + self.plot_height - (n + 1) * 22 - 20,
                                    LEFT_BORDER + 120,
                                    TOP_BORDER + self.plot_height - 20,
                                    fill='white', tags='axis')
            for i in xrange(n):
                self.c.create_text(LEFT_BORDER + 10,
                                   TOP_BORDER + self.plot_height + (i - n) * 22 - 20,
                                   fill=GRAPH_COLORS[i], tags='axis',
                                   text=self.legend[i], anchor=tk.W)

class GuiBarPlot(GuiPlot):

    def __init__(self, master, x=10, y=10, w=400, h=400, labels=None):
        GuiPlot.__init__(self, master, x, y, w, h, labels=labels)
        self.title('Barplot')

    def close_window(self):
        self.master.mybarplot = None
        self.destroy()

    def redraw(self):
        # remove old graph
        self.c.delete('graph')

        # add barplot
        n = self.values.shape[0]
        box_width = self.plot_width / float(n)
        for i, x in enumerate(numpy.linspace(box_width, self.plot_width, n)):
            self.c.create_rectangle(LEFT_BORDER + x - box_width,
                               TOP_BORDER + (1.0 - self.values[i]) * self.plot_height,
                               LEFT_BORDER + x,
                               TOP_BORDER + self.plot_height, fill=BARPLOT_COLORS[0], tags='graph')

        self.c.update()

class GuiGraph(GuiPlot):

    def __init__(self, master, x=10, y=10, w=400, h=400, legend=None):
        GuiPlot.__init__(self, master, x, y, w, h, legend=legend)
        self.title('Graph')

    def close_window(self):
        self.master.mygraph = None
        self.destroy()

    def redraw(self):
        # remove old graph
        self.c.delete('graph')

        # add graphs
        for i, graph in enumerate(self.values):
            n = len(graph)
            box_width = self.plot_width / float(n)
            graph = [1.0] + [x for x in graph]
            for j, x in enumerate(numpy.linspace(box_width, self.plot_width, n)):
                self.c.create_line(LEFT_BORDER + x - box_width,
                                   TOP_BORDER + (1.0 - graph[j]) * self.plot_height,
                                   LEFT_BORDER + x,
                                   TOP_BORDER + (1.0 - graph[j + 1]) * self.plot_height,
                                   fill=GRAPH_COLORS[i], tags='graph', width=2)
                self.c.create_text(LEFT_BORDER + x,
                                   TOP_BORDER + (1.0 - graph[j + 1]) * self.plot_height,
                                   fill=GRAPH_COLORS[i], tags='graph',
                                   text=GRAPH_PLOTCHARS[i], anchor=tk.CENTER)
        self.c.update()


def plot_R(v, verbose=True):
    """ 
        Create pdf-boxplots from run files.
        \param v parameters
    """

    if not os.path.isfile(os.path.join(v['RUN_FOLDER'], 'result.csv')):
        print 'No file %s found.' % os.path.join(v['RUN_FOLDER'], 'result.csv')
        sys.exit(2)
    result_file = open(os.path.join(v['RUN_FOLDER'], 'result.csv'), 'rU')
    reader = csv.reader(result_file, delimiter=',')

    # Read header names.
    data_header = reader.next()
    d = data_header.index('LOG_FILE')
    if v['EVAL_NAMES']: v['EVAL_NAMES'] = data_header[:d]
    else:               v['EVAL_NAMES'] = range(1, d + 1)

    # Prepare header.
    config = dict(TIME=[], NO_EVALS=[], LENGTH=[], NO_MOVES=[], ACC_RATE=[])
    for key in config.keys():
        try:    config[key].append(data_header.index(key))
        except: config[key].append(-1)
        config[key].append(0.0)

    # Read data.
    X = list()
    for i, row in enumerate(reader):
        if i == v['EVAL_MAX_DATA']: break
        X += [numpy.array([float(x) for x in row[:d]])]
        for key in config.keys():
            if config[key][0] > -1: config[key][1] += float(row[config[key][0]])
    result_file.close()
    X = numpy.array(X)

    if len(X) == 0:
        print 'Empty file %s.' % os.path.join(v['RUN_FOLDER'], 'result.csv')
        sys.exit(0)

    # Read effects.
    data_filename = os.path.join(v['SYS_ROOT'], v['DATA_PATH'], v['DATA_DATA_FILE'])
    if not data_filename[-4:].lower() == '.csv': data_filename += '.csv'
    effect_filename = os.path.join(data_filename.replace('.csv', '_effects.csv'))
    effects = list()
    if os.path.isfile(effect_filename):
        effect_file = open(effect_filename, 'rU')
        reader = csv.reader(effect_file, delimiter=',')
        reader.next() # skip header
        for row in reader:
            try:
                if not float(row[1]) == 0:
                    effects.append(numpy.array([data_header.index(row[0]) + 1, float(row[1])]))
            except ValueError:
                pass
        effect_file.close()
        # normalize effects
        effects = numpy.array(effects)
        effects[:, 1] /= numpy.abs(effects[:, 1]).max()


    # Compute averages.
    n = X.shape[0]
    d = X.shape[1]
    for key in config.keys(): config[key] = config[key][1] / float(n)
    config['TIME'] = str(datetime.timedelta(seconds=int(config['TIME'])))
    v.update(config)

    # Compute quantiles for the box plot.
    A = numpy.zeros((5, d))
    box = v['EVAL_BOXPLOT']
    X.sort(axis=0)
    for i in xrange(d):
        A[0][i] = X[:, i][0]
        for j, q in [(1, 1.0 - box), (2, 0.5), (3, box), (4, 1.0)]:
            A[j][i] = X[:, i][int(q * n) - 1] - A[:j + 1, i].sum()

    # Determine algorithm (might have been set on command line)
    algo = v['RUN_ALGO'].__name__
    if v['EVAL_FILE'][-8:] == 'mcmc.ini': algo = 'mcmc'
    if v['EVAL_FILE'][-9:] == 'amcmc.ini': algo = 'amcmc'

    colors = {'smc':'gold', 'mcmc':'firebrick', 'amcmc':'skyblue'}
    if v['EVAL_COLOR'] is None: v['EVAL_COLOR'] = colors[algo] + ['1', '3'][v['DATA_MAIN_EFFECTS']]

    # Format title.   
    title = 'ALGO %s, DATA %s, POSTERIOR %s, DIM %i, RUNS %i, TIME %s, NO_EVALS %.1f' % \
            (algo, v['DATA_DATA_FILE'], v['PRIOR_CRITERION'], d, n, v['TIME'], v['NO_EVALS'])
    if config['LENGTH'] > 0:
        title += '\nKERNEL %s, LENGTH %.1f, NO_MOVES %.1f, ACC_RATE %.3f' % \
            (v['MCMC_KERNEL'].__name__, v['LENGTH'], v['NO_MOVES'], v['ACC_RATE'])
    if verbose: print title + '\n'

    # Auto-adjust width
    if v['EVAL_WIDTH'] is None: v['EVAL_WIDTH'] = d * 0.1

    # Format dictionary.
    v.update({'EVAL_BOXPLOT':', '.join(['%.6f' % x for x in numpy.reshape(A, (5 * d,))]),
              'EVAL_EFFECTS':', '.join(['%d, %.6f' % (i, x) for (i, x) in effects]),
              'EVAL_DIM':str(d), 'EVAL_XAXS':A.shape[1] * 1.2 + 1,
              'EVAL_PDF':os.path.join(v['RUN_FOLDER'], 'plot.pdf').replace('\\', '/'),
              'EVAL_TITLE_TEXT': title,
              'EVAL_TITLE_SIZE': v['EVAL_TITLE'] * [v['EVAL_WIDTH'] * 10.0 / float(len(title)), 0.75][v['EVAL_LINES'] > 1]
              })

    # Format names.
    name_length = 0
    for i, x in enumerate(v['EVAL_NAMES']):
        if 'POW2' in x: v['EVAL_NAMES'][i] = x[:x.index('.')] + '.x.' + x[:x.index('.')]
        if name_length < len(str(x)):name_length = len(str(x))
    v['EVAL_NAMES'] = ', '.join(["'" + str(x).upper().replace('.X.', '.x.') + "'" for x in v['EVAL_NAMES']])

    # Auto-adjust margins
    if v['EVAL_INNER_MARGIN'] is None:
        v['EVAL_INNER_MARGIN'] = [name_length * 0.5, 2, 0.5, 0]
    if v['EVAL_OUTER_MARGIN'] is None:
        v['EVAL_OUTER_MARGIN'] = [0, 0, max(0.5, 2 * v['EVAL_TITLE_SIZE']), 0]
    for key in ['EVAL_OUTER_MARGIN', 'EVAL_INNER_MARGIN']:
        v[key] = ', '.join([str(x) for x in v[key]])

    # Create R-script.
    if v['EVAL_LINES'] > 1:
        v['EVAL_WIDTH'], v['EVAL_HEIGHT'] = 20, 20
        R = R_TEMPLATE.replace("pdf(", "pdf(paper='a4', ") % v
    else:
        R = R_TEMPLATE % v

    if v['EVAL_TITLE']:
        R += ("mtext('%(EVAL_TITLE_TEXT)s', family='%(EVAL_FONT_FAMILY)s', " +
                  "line=0.5, cex=%(EVAL_TITLE_SIZE)s, outer=TRUE)\n") % v

    R += 'dev.off()'
    R_file = open(os.path.join(v['SYS_ROOT'], v['RUN_PATH'], v['RUN_NAME'], 'plot.R'), 'w')
    R_file.write(R)
    R_file.close()

    # Execute R-script.
    subprocess.Popen([v['SYS_R'], 'CMD', 'BATCH', '--vanilla',
                      os.path.join(v['RUN_FOLDER'], 'plot.R'),
                      os.path.join(v['RUN_FOLDER'], 'plot.Rout')]).wait()

R_TEMPLATE = """
#
# This file was automatically generated.
#

# boxplot data from repeated runs
boxplot = c(%(EVAL_BOXPLOT)s)
boxplot = t(array(boxplot,c(length(boxplot)/5,5)))

# positions of effects
effects = c(%(EVAL_EFFECTS)s)
if (length(effects) > 0) effects = t(array(effects,c(2,length(effects)/2)))

# covariate names
names = c(%(EVAL_NAMES)s)

no_lines=%(EVAL_LINES)s
no_bars=ceiling(length(names)/no_lines)

# create PDF-file
pdf(file='%(EVAL_PDF)s', height=%(EVAL_HEIGHT)s, width=%(EVAL_WIDTH)s)
par(mfrow=c(no_lines,1), oma=c(%(EVAL_OUTER_MARGIN)s), mar=c(%(EVAL_INNER_MARGIN)s))

# create empty vector
empty=rep(0,length(names))

for(i in 1:no_lines) {
  start= (i-1)*no_bars + 1
  end  = min(i*no_bars, length(names))

  # create empty plot
  barplot(empty[start:end], ylim=c(0, 1), axes=FALSE, xaxs='i', xlim=c(-1, %(EVAL_XAXS)s/no_lines))
  
  # plot effects
  if (length(effects) > 0) {
      for (i in 1:dim(effects)[1]) {
        if (start <= effects[i,1] && effects[i,1] <= end) {
          empty[effects[i,1]] = 1.05
          barplot(empty[start:end], col=rgb(1,1-abs(effects[i,2]),1-abs(effects[i,2])), axes=FALSE, add=TRUE)
          barplot(empty[start:end], col='black', axes=FALSE, angle=sign(effects[i,2])*45, density=15, add=TRUE)
          empty[effects[i,1]]=0
        }
      }
  }
  
  # plot results
  barplot(boxplot[,start:end], ylim=c(0, 1), names=names[start:end], las=2, cex.names=0.5, cex.axis=0.75,
          axes=TRUE, col=c('%(EVAL_COLOR)s','black','white','white','black'), add=TRUE)
}
"""
