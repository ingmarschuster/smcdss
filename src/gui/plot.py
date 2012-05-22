# -*- coding: utf-8 -*-

"""
    GUI plots.
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
LEFT_BORDER = 45
RIGHT_BORDER = 10
GRAPH_PLOTCHARS = ['x', 'o', '']
GRAPH_COLORS = ['#990000', '#009900', '#5555EE']
CHAR_HEIGHT = 12.0
CHAR_WIDTH = 6.0
GREEK_RHO = u'\u03C1'

class GuiPlot(tk.Toplevel, object):

    name = 'Plot'

    def __init__(self, master, x=10, y=10, w=400, h=400, labels=None, legend=None):

        tk.Toplevel.__init__(self, master, height=h, width=w)
        self.geometry('+%d+%d' % (x, y))
        self.title(self.name + ': ' + master.myconfig['run/name']
                   + ' - ' + master.myconfig['prior/model']
                   + ' using ' + master.myconfig['prior/criterion'])

        self.legend = legend

        self.top_scale = 1.0
        self.values = None
        self.pressed_ctrl = False
        self.pressed_alt = False
        self.wheel_event = False

        # compute space for labels
        self.labels = labels

        # setup scrolled canvas
        self.sc = pmw.ScrolledCanvas(self, borderframe=1)
        self.c = self.sc._canvas
        self.c.configure(width=w, height=h, background='white')
        self.sc.pack(fill='both', expand=1)

        # initialize empty plot
        self.plot_width = w - LEFT_BORDER - RIGHT_BORDER
        self.compute_label_height()
        self.plot_height = h - TOP_BORDER - BOTTOM_BORDER - self.label_height

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

    def compute_label_height(self):
        if self.labels is None:
            self.label_height = 0
            return
        if self.plot_width == 0: return
        box_width = max(1.0, self.plot_width / float(len(self.labels) * CHAR_WIDTH))
        self.label_height = max([1.0] + [len(str(label)) / box_width for label in self.labels]) * CHAR_HEIGHT

    def set_pressed_ctrl(self, mode):
        self.pressed_ctrl = mode

    def set_pressed_alt(self, mode):
        self.pressed_alt = mode

    def wheel_handler(self, change):
        self.wheel_event = True
        if self.pressed_ctrl:
            self.plot_width -= change * 25
            self.plot_height += self.label_height
            self.compute_label_height()
            self.plot_height -= self.label_height
            self.draw_axis()
            self.redraw()
            self.sc.resizescrollregion()
        elif self.pressed_alt:
            self.c.xview_scroll(change, 'units')
        else:
            self.c.yview_scroll(change, 'units')

    def resize_window_handler(self, event):
        if self.pressed_ctrl and not self.wheel_event:
            self.c.bind('<Configure>', None)
            self.resize(event.width, event.height)
            self.c.bind('<Configure>', self.resize_window_handler)
        self.wheel_event = False

    def resize(self, width, height):
        self.plot_width = width - LEFT_BORDER - RIGHT_BORDER
        self.compute_label_height()
        self.plot_height = height - TOP_BORDER - BOTTOM_BORDER - self.label_height
        self.draw_axis()
        self.redraw()
        self.sc.resizescrollregion()

    def redraw(self):
        self.c.update()

    def draw_axis(self):
        # remove old axis
        self.c.delete('axis')

        # add labels
        if not self.labels is None:
            n = len(self.labels)
            box_width = self.plot_width / float(n)
            for i, x in enumerate(numpy.linspace(box_width, self.plot_width, n)):
                self.c.create_text(LEFT_BORDER + x - box_width / 2.0,
                              TOP_BORDER + self.plot_height + BOTTOM_BORDER - 8,
                              text=self.labels[i], tag='axis', width=box_width, font=('system', 6),
                              anchor=tk.N)

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
            self.c.create_text(1, TOP_BORDER + y, text='%.2f' % (self.top_scale * (1.0 - i / 10.0)), anchor=tk.W, tag='axis')

    def draw_legend(self):
        '''
            Add legend.
        '''
        if not self.legend is None:
            n = len(self.legend)
            self.c.create_rectangle(LEFT_BORDER + 5,
                                    TOP_BORDER + self.plot_height - (n + 1) * 22 + 10,
                                    LEFT_BORDER + 120,
                                    TOP_BORDER + self.plot_height - 10,
                                    fill='white', tags='axis')
            for i in xrange(n):
                self.c.create_text(LEFT_BORDER + 10,
                                   TOP_BORDER + self.plot_height + (i - n) * 22,
                                   fill=GRAPH_COLORS[i], tags='axis',
                                   text=self.legend[i], anchor=tk.W)

class GuiBarPlot(GuiPlot):

    name = 'Barplot'

    def __init__(self, master, x=10, y=10, w=400, h=400, labels=None, bar_color='#33ffff'):

        super(GuiBarPlot, self).__init__(master, x, y, w, h, labels=labels)
        self.bar_color = bar_color

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
                               TOP_BORDER + self.plot_height, fill=self.bar_color, tags='graph')
        self.c.update()


class GuiGraph(GuiPlot):

    name = 'Graph'

    def __init__(self, master, x=10, y=10, w=400, h=400, legend=None, adjust_scale=False, first_val=1.0):
        super(GuiGraph, self).__init__(master, x, y, w, h, legend=legend)
        self.lines = None
        self.first_val = first_val
        self.adjust_scale = adjust_scale

    def close_window(self):
        self.master.mygraph = None
        self.destroy()

    def redraw(self):
        # remove old graph
        self.c.delete('graph')

        if self.adjust_scale:
            flat_list = [item for sublist in self.values for item in sublist]
            if len(flat_list) == 0: top_scale = 1.0
            else:
                top_scale = round(max(flat_list) + 0.05, 1)
            if not top_scale == self.top_scale:
                self.top_scale = top_scale
                self.draw_axis()

        # add lines
        if not self.lines is None:
            for j, graph in enumerate(self.lines):
                n = sum(graph)
                if n == 0: break
                box_width = self.plot_width / float(n)
                for k in numpy.cumsum(numpy.array(graph)):
                    self.c.create_line(LEFT_BORDER + k * box_width,
                                       TOP_BORDER + self.plot_height,
                                       LEFT_BORDER + k * box_width,
                                       TOP_BORDER,
                                       fill=GRAPH_COLORS[-1], tags='graph', width=2)
        self.draw_legend()

        # add graphs
        for i, graph in enumerate(self.values):
            n = len(graph)
            if n == 0: break
            box_width = self.plot_width / float(n)
            graph = [self.first_val] + [x for x in graph]
            for j, x in enumerate(numpy.linspace(box_width, self.plot_width, n)):
                self.c.create_line(LEFT_BORDER + x - box_width,
                                   TOP_BORDER + (self.top_scale - graph[j]) * self.plot_height / self.top_scale,
                                   LEFT_BORDER + x,
                                   TOP_BORDER + (self.top_scale - graph[j + 1]) * self.plot_height / self.top_scale,
                                   fill=GRAPH_COLORS[i], tags='graph', width=2)
                self.c.create_text(LEFT_BORDER + x,
                                   TOP_BORDER + (self.top_scale - graph[j + 1]) * self.plot_height / self.top_scale,
                                   fill=GRAPH_COLORS[i], tags='graph',
                                   text=GRAPH_PLOTCHARS[i], anchor=tk.CENTER)
        self.c.update()

def plot_R(myconfig):
    """ 
        Create pdf-boxplots from run files.
        \param v parameters
    """

    if not os.path.isfile(os.path.join(myconfig['run/folder'], 'result.csv')):
        sys.stdout.write('\rNo file %s found.' % os.path.join(myconfig['run/folder'], 'result.csv'))
        return False
    result_file = open(os.path.join(myconfig['run/folder'], 'result.csv'), 'rU')
    reader = csv.reader(result_file, delimiter=',')

    # Read header names.
    data_header = reader.next()
    d = data_header.index('LOG_FILE')
    if myconfig['layout/names']:
        myconfig['layout/names'] = data_header[:d]
    else:
        myconfig['layout/names'] = range(1, d + 1)

    # Prepare header.
    config = dict(TIME=[], NO_EVALS=[], LENGTH=[], NO_MOVES=[], ACC_RATE=[])
    for key in config.keys():
        try:    config[key].append(data_header.index(key))
        except: config[key].append(-1)
        config[key].append(0.0)

    # Read data.
    X = list()
    for i, row in enumerate(reader):
        if i == myconfig['layout/max_data']: break
        X += [numpy.array([float(x) for x in row[:d]])]
        for key in config.keys():
            if config[key][0] > -1: config[key][1] += float(row[config[key][0]])
    result_file.close()
    X = numpy.array(X)

    if len(X) == 0:
        print '\rEmpty file %s.' % os.path.join(myconfig['run/folder'], 'result.csv')
        return False

    # Read effects.
    data_filename = os.path.join(myconfig['path/root'], myconfig['path/data'], myconfig['data/csv_file'])
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
    myconfig.update(config)

    # Compute quantiles for the box plot.
    A = numpy.zeros((5, d))
    box = myconfig['layout/boxplot']
    X.sort(axis=0)
    for i in xrange(d):
        A[0][i] = X[:, i][0]
        for j, q in [(1, 1.0 - box), (2, 0.5), (3, box), (4, 1.0)]:
            A[j][i] = X[:, i][int(q * n) - 1] - A[:j + 1, i].sum()

    colors = {'smc':'gold', 'mcmc':'firebrick', 'amcmc':'skyblue'}
    if myconfig['layout/color'] is None:
        myconfig['layout/color'] = colors[myconfig['run/algo']] + ['1', '3'][myconfig['data/main_effects']]

    # Format title.   
    title = 'ALGO %s, DATA %s, POSTERIOR %s, DIM %i, RUNS %i, TIME %s, NO_EVALS %.1f' % \
            (myconfig['run/algo'], myconfig['data/csv_file'], myconfig['prior/criterion'],
             d, n, myconfig['TIME'], myconfig['NO_EVALS'])
    if config['LENGTH'] > 0:
        title += '\nKERNEL %s, LENGTH %.1f, NO_MOVES %.1f, ACC_RATE %.3f' % \
            (myconfig['mcmc/kernel'].__name__, myconfig['length'], myconfig['no_moves'], myconfig['acc_rate'])

    # Auto-adjust width
    if myconfig['layout/width'] is None: myconfig['layout/width'] = d * 0.1

    # Format dictionary.
    myconfig.update({'layout/data':', '.join(['%.6f' % x for x in numpy.reshape(A, (5 * d,))]),
          'layout/effects':', '.join(['%d, %.6f' % (i, x) for (i, x) in effects]),
          'layout/dim':str(d), 'layout/xaxs':A.shape[1] * 1.2 + 1,
          'layout/pdf':os.path.join(myconfig['run/folder'], 'plot.pdf').replace('\\', '/'),
          'layout/title text': title,
          'layout/title_size': myconfig['layout/title'] * \
                [myconfig['layout/width'] * 10.0 / float(len(title)), 0.75][myconfig['layout/lines'] > 1]
                })

    # Format names.
    name_length = 0
    for i, x in enumerate(myconfig['layout/names']):
        if 'POW2' in x: myconfig['layout/names'][i] = x[:x.index('.')] + '.x.' + x[:x.index('.')]
        if name_length < len(str(x)) + 1:
            name_length = len(str(x)) + 1
    myconfig['layout/names'] = ', '.join(["'" + str(x).upper().replace('.X.', '.x.')
                                        + "'" for x in myconfig['layout/names']])

    # Auto-adjust margins
    if myconfig['layout/inner_margin'] is None:
        myconfig['layout/inner_margin'] = [name_length * 0.6, 2, 0.5, 0]
    if myconfig['layout/outer_margin'] is None:
        myconfig['layout/outer_margin'] = [0, 0, max(0.5, 2 * myconfig['layout/title_size']), 0]
    for key in ['layout/outer_margin', 'layout/inner_margin']:
        if not isinstance(myconfig[key], str):
            myconfig[key] = ', '.join([str(x) for x in myconfig[key]])

    # Create R-script.
    if myconfig['layout/lines'] > 1:
        myconfig['layout/width'], myconfig['layout/height'] = 20, 20
        R = R_TEMPLATE.replace("pdf(", "pdf(paper='a4', ") % myconfig
    else:
        R = R_TEMPLATE % myconfig

    if myconfig['layout/title']:
        R += ("mtext('%(layout/title_text)s', family='%(layout/font_family)s', " +
                  "line=0.5, cex=%(layout/title_size)s, outer=TRUE)\n") % myconfig

    R += 'dev.off()'
    R_file = open(os.path.join(myconfig['path/root'], myconfig['path/run'], myconfig['run/name'], 'plot.R'), 'w')
    R_file.write(R)
    R_file.close()

    # Execute R-script.
    subprocess.Popen([myconfig['path/r'], 'CMD', 'BATCH', '--vanilla',
                      os.path.join(myconfig['run/folder'], 'plot.R'),
                      os.path.join(myconfig['run/folder'], 'plot.Rout')]).wait()

R_TEMPLATE = """
#
# This file was automatically generated.
#

# boxplot data from repeated runs
data = c(%(layout/data)s)
data = t(array(data,c(length(data)/5,5)))

# positions of effects
effects = c(%(layout/effects)s)
if (length(effects) > 0) effects = t(array(effects,c(2,length(effects)/2)))

# covariate names
names = c(%(layout/names)s)

no_lines=%(layout/lines)s
no_bars=ceiling(length(names)/no_lines)

# create PDF-file
pdf(file='%(layout/pdf)s', height=%(layout/height)s, width=%(layout/width)s)
par(mfrow=c(no_lines,1), oma=c(%(layout/outer_margin)s), mar=c(%(layout/inner_margin)s))

# create empty vector
empty=rep(0,length(names))

for(i in 1:no_lines) {
  start= (i-1)*no_bars + 1
  end  = min(i*no_bars, length(names))

  # create empty plot
  barplot(empty[start:end], ylim=c(0, 1), axes=FALSE, xaxs='i', xlim=c(-1, %(layout/xaxs)s/no_lines))
  
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
  barplot(data[,start:end], ylim=c(0, 1), names=names[start:end], las=2, cex.names=0.5, cex.axis=0.75,
          axes=TRUE, col=c('%(layout/color)s','black','white','white','black'), add=TRUE)
}
"""
