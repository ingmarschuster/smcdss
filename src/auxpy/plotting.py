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

def plot4(f, path, models=None):
    '''
        Compares f to its binary model approximations. Generates a pseudo sample from f to initialize the binary models.
        Plots the true f and histograms obtained from the models. Works only for dimensions up to 5.  
        @param f target function
        @param outfile output file
        @param models list of binary models
    '''

    m = 5000 # number of pseudo samples from f
    n = 5000 # number of random draws from models
    d = f.d
    if models is None: models = [ProductBinary, LogisticRegrBinary]

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
    ymax = hist.max() + 0.1

    color = ['grey75', 'black', 'blue']

    for index in range(z):

        # plot with rpy
        r.pdf(file=path + '/out' + str(index) + '.pdf')
        r.par(oma=[0, 0, 0, 0], family="serif")
        r.barplot(hist[index], ylim=[0, ymax], \
                  names=names, cex_names=1.5, las=3, \
                  col=color[index], family="serif", cex_axis=1.5)
        r.dev_off()

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
    plot4(f=f, path='../../data/testruns/toy')
    
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
