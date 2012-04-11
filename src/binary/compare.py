#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Runs and evaluates a comparison of parametric families for sampling multivariate
binary data with given correlations.

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
@endverbatim
"""

"""
\namespace binary.compare
"""

from binary.base import moments2corr, corr2moments, random_moments
from binary.gaussian_copula import GaussianCopulaBinary
from binary.student_copula import StudentCopulaBinary
from binary.linear_cond import LinearCondBinary
from binary.logistic_cond import LogisticCondBinary

import utils
import getopt
import numpy
import os
import scipy.linalg
import subprocess
import sys

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
        f.write('rho,student,gaussian,logistic,linear\n')
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
        f_name = os.path.expanduser('~/Documents/Data/bg/test_%d.pdf' % d)
        if not os.path.isfile(f_name):
            plot(d=d)
        if os.name == 'posix':
            viewer = 'okular'
        else:
            viewer = os.path.expanduser('~/Documents/Software/portable/viewer/PDFXCview')
        subprocess.Popen([viewer, f_name])


def plot(d):
    cbg_dir = os.path.expanduser('~/Documents/Data/bg')
    if os.name == 'posix':
        R = 'R'
    else:
        R = os.path.expanduser('~/Documents/Software/portable/r/App/R-2.11.1/bin/R.exe')
    subprocess.Popen([R, 'CMD', 'BATCH', '--vanilla', '--args d=%d' % d,
                      os.path.join(cbg_dir, 'plot.R'),
                      os.path.join(cbg_dir, 'plot.Rout')]).wait()


def compare(d, ticks=15, n=1e6):

    eps = 0.01
    delta = 0.005
    if d < 15: delta = 0.0
    score = numpy.zeros((ticks + 1, 5), dtype=float)
    score[:, 0] = numpy.linspace(0.0, 1.0, ticks + 1)
    norm = lambda x: scipy.linalg.norm(x, ord=2)

    for i in xrange(ticks + 1):

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
        loss['student'] = M - corr2moments(generator.mean, generator.corr)
        generator = GaussianCopulaBinary.from_moments(mean, corr, delta=delta)
        loss['gaussian'] = M - corr2moments(generator.mean, generator.corr)
        generator = LogisticCondBinary.from_moments(mean, corr, delta=delta, verbose=False)
        loss['logistic'] = M - corr2moments(*generator.rvs_marginals(n, 1))
        generator = LinearCondBinary.from_moments(mean, corr)
        loss['linear'] = M - corr2moments(*generator.rvs_marginals(n, 1))

        ref = loss['product']
        ref = ref * (numpy.abs(ref) > eps)
        for k, generator in enumerate(['student', 'gaussian', 'logistic', 'linear']):
            gen = loss[generator]
            gen = gen * (numpy.abs(gen) > eps)
            score[i, k + 1] = (norm(ref) - norm(gen)) / norm(ref)

    return score

if __name__ == "__main__":
    main()
