#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Cross-moment comparison of parametric families.

@verbatim
USAGE:
        cbg [option]

OPTIONS:
        -d    dimension
        -r    run comparison
        -c    start clean
        -e    evaluate results
        -v    open plot with standard viewer
        -m    start multiple processes
        -n    number of samples
        -t    numer of ticks
@endverbatim
"""

"""
@namespace binary.compare.moments
"""

from __init__ import *
from binary.base import moments2corr, corr2moments, random_moments
import getopt
import numpy
import os
import scipy.linalg
import subprocess
import sys
import utils.auxi

generators = ['cop_student', 'cop_gaussian', 'cond_logistic', 'cond_arctan', 'cond_linear']

def main():
    """ Main method. """

    # Parse command line options.
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'r:d:ecvm:t:n:')
    except getopt.error, msg:
        print msg
        sys.exit(2)

    # Check arguments and options.
    if len(opts) == 0:
        print __doc__.replace('@verbatim', '').replace('@endverbatim', '')
        sys.exit(0)

    m, r, e, c, v, d, n, t = 0, 0, False, False, False, None, 1e6, 15

    # Start multiple processes.
    for o, a in opts:
        if o == '-m': m = int(a)
        if o == '-r': r = int(a)
        if o == '-d': d = int(a)
        if o == '-e': e = True
        if o == '-v': v = True
        if o == '-c': c = True
        if o == '-n': n = float(a)
        if o == '-t': t = int(a)

    if d is None:
        print __doc__.replace('@verbatim', '').replace('@endverbatim', '')
        print '\n stopped. no dimension given.'
        sys.exit(0)

    # prepare output file
    f_name = os.path.expanduser('~/Documents/Data/bg/test_%d.csv' % d)
    if not os.path.isfile(f_name) or c:
        f = open(f_name, 'w')
        f.write('rho,%s\n' % ','.join(generators))
        f.close()

    # start external processes
    while m > 0:
        if os.name == 'posix':
            subprocess.call('gnome-terminal -e "cbg ' + ' '.join([o + ' ' + a for (o, a) in opts if not o in ['-m', '-c']]) + '"', shell=True)
        else:
            path = os.path.abspath(os.path.join(os.path.join(*([os.getcwd()] + ['..'] * 1)), 'bin', 'cbg.bat'))
            subprocess.call('start "cbg" /MAX "%s" ' % path + ' '.join([o + ' ' + a for (o, a) in opts if not o in ['-m', '-c']]), shell=True)
        m -= 1
        r = 0

    # start runs
    r_all = r
    if r > 0: print 'ticks: %d, samples: %.f' % (t, n)
    while r > 0:
        print 'start %d/%d' % (r_all - r + 1, r_all)
        f_name = os.path.expanduser('~/Documents/Data/bg/test_%d.csv' % d)
        res = compare(d, ticks=t, n=n)
        csv = '\n'.join([','.join(['%.8f' % col for col in row]) for row in res]) + '\n'
        # write to file
        f = open(f_name, 'a')
        f.write(csv)
        f.close()
        print '\n'
        r -= 1

    # launch plotter
    if e:
        plot(d=d)

    # launch viewer
    if v:
        if os.name == 'posix':
            viewer = 'okular'
        else:
            viewer = os.path.expanduser('~/Documents/Software/portable/viewer/PDFXCview')
        for filename in os.listdir(os.path.expanduser('~/Documents/Data/bg')):
            if not filename.endswith('_%d.pdf' % d): continue
            subprocess.Popen([viewer, os.path.expanduser('~/Documents/Data/bg/%s' % filename)])


def compare(d, ticks=15, n=1e6):

    eps = 0.01
    delta = 0.005
    if d < 15: delta = 0.0
    score = numpy.zeros((ticks + 1, len(generators) + 1), dtype=float)
    score[0, 1:] = 1.0
    score[1:, 0] = numpy.linspace(0.0, 1.0, ticks + 1)[1:]
    norm = lambda x: scipy.linalg.norm(x, ord=2)
    utils.auxi.progress(0.0)

    for i in xrange(1, ticks + 1):

        utils.auxi.progress(score[i, 0])

        # sample random moments
        mean, corr = moments2corr(random_moments(d, rho=score[i, 0]))
        M = corr2moments(mean, corr)
        M_star = numpy.outer(mean, mean)
        M_star = numpy.triu(M_star, 1) + numpy.tril(M_star, -1) + numpy.diag(mean)
        loss = {}

        # compute parametric families and reference loss
        loss['product'] = M - M_star
        generator = StudentCopulaBinary.from_moments(mean, corr, delta=delta)
        loss['cop_student'] = M - corr2moments(generator.mean, generator.corr)
        generator = GaussianCopulaBinary.from_moments(mean, corr, delta=delta)
        loss['cop_gaussian'] = M - corr2moments(generator.mean, generator.corr)
        generator = LogisticCondBinary.from_moments(mean, corr, delta=delta)
        loss['cond_arctan'] = M - corr2moments(*generator.rvs_marginals(n, 1))
        generator = ArctanCondBinary.from_moments(mean, corr, delta=delta)
        loss['cond_logistic'] = M - corr2moments(*generator.rvs_marginals(n, 1))
        generator = LinearCondBinary.from_moments(mean, corr)
        loss['cond_linear'] = M - corr2moments(*generator.rvs_marginals(n, 1))

        ref = loss['product']
        ref = ref * (numpy.abs(ref) > eps)
        for k, generator in enumerate(generators):
            gen = loss[generator]
            gen = gen * (numpy.abs(gen) > eps)
            score[i, k + 1] = (norm(ref) - norm(gen)) / norm(ref)

    return score


def plot(d):
    cbg_dir = os.path.expanduser('~/Documents/Data/bg')
    if os.name == 'posix':
        R = 'R'
    else:
        R = os.path.expanduser('~/Documents/Software/portable/r/App/R-2.11.1/bin/R.exe')
    subprocess.Popen([R, 'CMD', 'BATCH', '--vanilla', '--args d=%d' % d,
                      os.path.join(cbg_dir, 'plot.R'),
                      os.path.join(cbg_dir, 'plot.Rout')]).wait()

if __name__ == "__main__":
    main()

"""

# auxiliary function for adding legend
legend.col <- function(colr, lev){

  opar <- par
  n <- length(colr)
  bx <- par("usr")
  box.cx <- c(bx[2] + (bx[2] - bx[1]) / 1000,
  bx[2] + (bx[2] - bx[1]) / 1000 + (bx[2] - bx[1]) / 50)
  box.cy <- c(bx[3], bx[3])
  box.sy <- (bx[4] - bx[3]) / n
  xx <- rep(box.cx, each = 2)
  par(xpd = TRUE)
  for(i in 1:n){
    yy <- c(box.cy[1] + (box.sy * (i - 1)),
    box.cy[1] + (box.sy * (i)),
    box.cy[1] + (box.sy * (i)),
    box.cy[1] + (box.sy * (i - 1)))
    polygon(xx, yy, col = colr[i], border = colr[i])
  }
  rect(0,0,box.cx,1.02, col=NA, border='black',lwd=1.2)
  rect(1,0,box.cx,1.02, col=NA, border='black',lwd=1.2)
  par(new = TRUE)
  plot(0, 0, type = "n", ylim = c(min(lev), max(lev)),
       yaxt = "n", ylab = "", xaxt=  "n", xlab = "", frame.plot = FALSE)
  axis(side=4, las=2, tick=FALSE, lwd=2)
  par <- opar
}


#First read in the arguments listed at the command line
args=(commandArgs(TRUE))

if(length(args)==0){
    print("No arguments supplied.")
    ##supply default values
    d = 3
}else{
    for(i in 1:length(args)){
         eval(parse(text=args[[i]]))
    }
}

csv_name=paste('~/Documents/Data/bg/test_',d,'.csv',sep='')

Z=read.csv(csv_name)
h=unique(Z$rho)
n=length(h)
colors = c('gold', 'skyblue', 'seagreen3', 'firebrick4', 'aquamarine2', 'lightpink3')
family = c('cond_arctan', 'qu_linear', 'cop_student', 'cop_gaussian', 'cond_logistic', 'cond_linear')

v = seq(0,1,length.out=n)
nshades=20
dist=seq(0.5,0,length.out=nshades)
shades=grey(seq(0.97, 0.1, length.out=nshades))

for (j in 1:length(family)) {

  # transform data  
  g=NULL
  for (i in 1:n) {
    g[i]=subset(Z,select=family[j], subset=Z[1]==h[i])
  }
  g=unlist(g)
  dim(g)=c(length(g)/n,n)
  
  # PDF
  pdf_name=paste('~/Documents/Data/bg/',family[j],'_',d,'.pdf',sep='')
  pdf(file=pdf_name, height=3, width=5)
  par(mar=c(2, 2.5, .6, 2.5), family='serif')

  # set up empty plot
  plot(0, xaxt='n', yaxt='n', xlim=c(0,1), ylim=c(0,1.02), yaxs="i", xaxs="i", type="n")
  
  # draw shades of quantiles
  for (i in 1:nshades) {
    q = apply(g, 2, function(x) quantile(x, .5+c(-dist[i], dist[i])))
    xx = c(v, v[n:1])
    yy = c(q[1,], rev(q[2,]))
    polygon(xx, yy, col=shades[i], border=shades[i])
  }

  # draw grid
  for (x in 1:15/15) {abline(v=x, lty=3, col='grey50')}
  for (y in 1:10/10) {abline(h=y, lty=3, col='grey50')}
  
  # draw axis
  axis(side=1,lwd=1.2)
  axis(side=2,lwd=1.2,las=1)

  # draw median
  m = apply(g, 2, median)
  lines(v, m, col=colors[j], lwd=2.5, type='b')
  
  # draw legend
  legend.col(col=shades, lev=dist)
  dev.off()
}

"""
