#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Toy example comparison of parametric families.
@namespace binary.analysis.plotter
"""

from __init__ import *
import binary.base as base
import numpy
import os

generator_colors = {
'product family':'skyblue',
'student copula family':'seagreen3',
'gaussian copula family':'seagreen3',
'logistic conditionals family':'gold',
'linear conditionals family':'gold',
'arctan conditionals family':'gold',
'quadratic linear family':'firebrick4'
}

generator_classes = [
ProductBinary,
StudentCopulaBinary,
GaussianCopulaBinary,
LogisticCondBinary,
LinearCondBinary,
QuLinearBinary]

def main():
    plot_toy_bvs()

def plot_toy_bvs():
    """
        Plots comparison based on Bayesian Variable Selection problem.
    """
    size = 100
    y, X = numpy.empty(size), numpy.empty((size, 4))
    for i in xrange(100):
        W = [numpy.random.normal() + mean for mean in [-10.0, +10.0]]
        Z = [x + 2.5 * numpy.random.normal() for x in 2 * W]
        y[i] = W[0] + W[1]
        X[i] = numpy.array(Z)
    config = {'data/static': 0,
             'prior/criterion':'bayes',
             'prior/var_dispersion':size,
             'prior/cov_matrix_hp':'independent',
             'prior/var_hp_a':0.0,
             'prior/var_hp_b':0.0,
             'prior/model_inclprob':0.5,
             'prior/model_maxsize':None,
             'data/constraints':[]
             }
    f = SelectorLnBayes(y=y, Z=X, config=config)
    f2R(f=f, path='~/Documents/R')

def plot_toy_ubqo():
    """
        Plots comparison based on Unconstrained Binary Quadratic Optimization problem.
    """
    vec = ''' 1,  2,  1,  0,
              2,  1, -3, -2,
              1, -3,  1,  2,
              0, -2,  2, -2'''
    A = numpy.array(eval('[' + vec + ']')).reshape((4, 4)) / 1.0
    f = QuExpBinary(A=A)
    mean, corr = f.exact_marginals ()
    print 'A:\n%s' % repr(A)
    print 'mean:\n%s' % repr(mean)
    print 'corr:\n%s' % repr(corr)
    f2R(f=f, path='~/Documents/R')

def f2R(f, path, m=5e4, n=5e4):
    """
        Compares f to its binary model approximations. Generates a pseudo sample
        from f to initialize the binary generators. Plots the true f and histograms
        obtained from the generators. Works only for dimensions up to 5.
        \param f target function
        \param outfile output file
        \param generators list of binary generators
    """

    d, n = f.d, int(n)
    path = os.path.expanduser(path)
    hist = numpy.array((len(generator_classes) + 1) * [numpy.zeros(2 ** d)])
    nfunc = len(generator_classes) + 1

    # explore posterior
    names = []
    for dec in range(2 ** d):
        b = base.dec2bin(dec, d)
        names.append(base.bin2str(b))
        hist[0, dec] = f.pmf(b)
    hist[0] /= hist[0].sum()

    # create pseudo sample
    X = list()
    for dec in range(2 ** d):
        for k in xrange(int(m * hist[0, dec])):
            X.append(base.dec2bin(dec, d))
    X = numpy.array(X)

    # init approximations
    generators = list()
    for generator in generator_classes:
        generators.append(generator.from_data(X=X, weights=None))

    generators.insert(0, f)
    for k in xrange(n):
        for index in range(1, nfunc):
            dec = base.bin2dec(generators[index].rvs())
            hist[index, dec] += 1
    hist[1:] /= float(n)
    ymax = hist.max() + 0.05
    hist_str = (['c(' + ', '.join(['%.6f' % x for x in hist[i]]) + ')' for i in range(nfunc)])

    pdfnames = ['function'] + [generator.name.replace('family', '').strip().replace(' ', '_') for generator in generator_classes]
    colors = ['violetred'] + [generator_colors[generator.name] for generator in generator_classes]

    args = dict(hist=',\n'.join(hist_str),
                names="', '".join(names),
                ymax=ymax,
                colors="', '".join(colors),
                pdfnames="', '".join(pdfnames),
                path=path + '/toy_plot/')
    R = """
    hist = list(\n%(hist)s)
    names = c('%(names)s')
    colors = c('%(colors)s')
    pdfnames = c('%(pdfnames)s')
    
    for (index in 1:length(pdfnames)) {
        pdf(file=paste('%(path)s', paste(pdfnames[index],'pdf',sep='.'), sep='/'), height=4, width=10)      
        par(oma=c(0, 0, 0, 0), mar=c(5.2, 3.8, 0, 0), family="serif")
        barplot(hist[[index]], ylim=c(0, 0.49), xlim=c(0, 16*1.05+0.05), space=0.05,
                names=names, cex.names=2.5, las=2, col=colors[index],
                family="serif", cex.axis=2.5, xaxs='i', border=TRUE)
        dev.off()
        
    } """ % args

    filename = open(os.path.join(path, 'toy.R'), 'w')
    filename.write(R)
    filename.close()
    cwd = os.getcwd()
    os.chdir(path)
    os.system('R --vanilla --slave  <' + os.path.join(path, 'toy.R'))
    os.chdir(cwd)

if __name__ == "__main__":
    main()
