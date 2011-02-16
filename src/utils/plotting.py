#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

from rpy import *
from data import *
from binary import *
from numpy import *

def plot4(f, outfile=None, models=None):
    '''
        Compares f to its binary model approximations. Generates a pseudo sample from f to initialize the binary models.
        Plots the true f and histograms obtained from the models. Works only for dimensions up to 5.  
        @param f target function
        @param outfile output file
        @param models list of binary models
    '''

    m = 10000 # number of pseudo samples from f
    n = 10000 # number of random draws from models
    d = f.d
    if outfile is None: outfile = f.dataFile[:-3] + 'pdf'
    if models is None: models = [LogisticRegrBinary, ProductBinary, HiddenNormalBinary]

    names = []
    hist = array(4 * [zeros(2 ** d)])

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
        for index in range(1, 4):
            dec = bin2dec(models[index].rvs())
            hist[index][dec] += 1
    hist[1:] /= float(n)
    ymax = hist.max()

    color = ['grey85', 'grey65', 'grey45', 'grey25']

    # plot with rpy
    r.pdf(paper="a4", file=outfile, width=12, height=12)
    r.par(mfrow=[2, 2], oma=[40, 4, 0, 4], mar=[1, 2, 4, 1])
    for index in range(4):
        r.barplot(hist[index], ylim=[0, ymax], \
                  names=names, cex_names=0.8, las=3, \
                  col=color[index], family="serif", cex_axis=0.8)
        r.title(main=models[index].name, line=1, family="serif", font_main=1, cex_main=1)
    r.mtext("Histograms of n=%i based on m=%i pseudo samples" % (n, m), outer=True, line=1, cex=1.5)
    r.dev_off()

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
    colors = map(_color, z.reshape(d*d,) )
    z = abs(z)
    r.library('gplots')
    r.pdf(paper="a4r", file='/home/cschafer/Bureau/balloon.pdf', width=12, height=12)
    r.par(mar=[0, 10, 0, 10])
    r.balloonplot(x, y, z, show_margins=False, cum_margins=False, label=False, label_lines=False,
                  xlab='', ylab='', zlab='', main='', dotcolor=colors, axes=False)
    r.dev_off()
