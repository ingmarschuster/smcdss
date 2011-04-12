#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Plotting via R.
"""

"""
@namespace utils.plotter
$Author$
$Rev$
$Date$
@details
"""

__version__ = "$Revision$"

import os
from numpy import *
from data import *
from binary import *

def plot4(f, path, models=None):
    """
        Compares f to its binary model approximations. Generates a pseudo sample from f to initialize the binary models.
        Plots the true f and histograms obtained from the models. Works only for dimensions up to 5.  
        @param f target function
        @param outfile output file
        @param models list of binary models
    """

    m = 5000 # number of pseudo samples from f
    n = 5000 # number of random draws from models
    d = f.d
    if models is None: models = [ProductBinary, LogisticBinary]

    names = []
    hist = array(3 * [zeros(2 ** d)])
    z = len(models) + 1

    # explore posterior
    sample = data()
    for dec in range(2 ** d):
        bin = dec2bin(dec, d)
        names.append(bin2str(bin))
        hist[0][dec] = f.pmf(bin)
        # create pseudo sample
        for j in range(int(m * hist[0][dec])):
            sample.append(bin)
    hist[0] /= hist[0].sum()

    # init approximations
    models = [model.from_data(sample) for model in models]
    models.insert(0, f)
    for k in range(n):
        for index in range(1, z):
            dec = bin2dec(models[index].rvs())
            hist[index][dec] += 1
    hist[1:] /= float(n)
    ymax = hist.max() + 0.05
    hist_str = (['c(' + ', '.join(['%.6f' % x for x in hist[i]]) + ')' for i in range(z)])

    args = dict(hist=',\n'.join(hist_str),
                names="', '".join(names),
                ymax=ymax,
                colors="', '".join(['grey75', 'black', 'blue']),
                pdfnames="', '".join(['function', 'product', 'logregr']))
    R = """
    hist = list(\n%(hist)s)
    names = c('%(names)s')
    colors = c('%(colors)s')
    pdfnames = c('%(pdfnames)s')
    
    for (index in 1:3) {
        pdf(file=paste('pdf', paste(pdfnames[index],'pdf',sep='.'), sep='/'), height=5, width=10)
        par(oma=c(0, 0, 0, 0), mar=c(3.5, 2.5, 0, 0), family="serif")
        barplot(hist[[index]], ylim=c(0, %(ymax).6f), names=names, cex.names=1.5, las=3, col=colors[index], family="serif", cex.axis=1.5)
        dev.off()
    } """ % args

    file = open(os.path.join(path, 'toy.R'), 'w')
    file.write(R)
    file.close()
    cwd = os.getcwd()
    os.chdir(path)
    os.system('R --vanilla --slave  <' + os.path.join(path, 'toy.R'))
    os.chdir(cwd)

class toy(object):
    def rvs(self):
        W = [random.normal() + mean for mean in [-10.0, +10.0]]
        X = [x + 5 * random.normal() for x in 2 * W]
        X.insert(0, W[0] + W[1])
        return array(X)

def plot_toy():
    d = data()
    t = toy()
    d.sample(q=t, size=100)
    f = PosteriorBinary(sample=d.X, posterior_type='hb')
    plot4(f=f, path='../../data/r')

plot_toy()

def _color(x):
    if x == 1.0: return 'grey25'
    if x > 0: return 'red'
    if x < 0: return 'skyblue'

def plot_matrix():
    d = 100
    x = array(d * range(1, d + 1))
    y = copy(x)
    y.sort()
    z = random.normal(size=(d, d))
    z = dot(z.T, z)
    v = diag(z)
    z /= sqrt(dot(v[:, newaxis], v[newaxis, :]))
    colors = map(_color, z.reshape(d * d,))
    z = abs(z)
    r.library('gplots')
    r.pdf(paper="a4r", file='/home/cschafer/Bureau/balloon.pdf', width=12, height=12)
    r.par(mar=[0, 10, 0, 10])
    r.balloonplot(x, y, z, show_margins=False, cum_margins=False, label=False, label_lines=False,
                  xlab='', ylab='', zlab='', main='', dotcolor=colors, axes=False)
    r.dev_off()