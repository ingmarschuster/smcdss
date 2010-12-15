#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-12-03 19:08:22 +0100 (ven., 03 déc. 2010) $
    $Revision: 38 $
'''

import sys, getopt, time, shutil, csv, os
import auxpy.logger
from auxpy.data import data
import binary
import default
import algos
from numpy.random import seed

try:    from rpy import *
except: pass

def main():

    param = {}
    param.update(default.param)
    param.update({'sys_path':sys.path[0][:-4]})

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hreocv')
    except getopt.error, msg:
        print msg
        sys.exit(2)

    if len(args) == 0: sys.exit(0)

    param.update({'test_name':os.path.splitext(os.path.basename(args[0]))[0]})
    param.update({'test_folder':os.path.join(param['sys_path'] , param['test_path'] , param['test_name'])})

    if not os.path.isfile(param['test_folder'] + '.py'):
        print 'The test file "' + param['test_name'] + '" does not exist in the test path.'
        sys.exit(0)

    sys.path.insert(0, os.path.join(param['sys_path'] , param['test_path']))
    user = __import__(param['test_name'])
    param.update(user.param)

    # process options
    opts = [o[0] for o in opts]
    if '-h' in opts: print __doc__
    if '-c' in opts:
        try:
            shutil.rmtree(param['test_folder'])
        except:
            pass
    if '-r' in opts: _testrun(param, True)
    if '-e' in opts: _eval_mean(param)
    if '-o' in opts: os.system('okular ' + param['test_folder'] + '/eval.pdf  &')


def _testrun(param, verbose=False):

    # seed(1)

    # read data
    data_file = os.path.join(param['sys_path'], param['data_path'], param['data_set'], param['data_set'] + '.out')
    reader = csv.reader(open(data_file, 'r'), delimiter='\t')
    n = min(param['data_n_covariates'] + 1, len(reader.next()))
    sample = array([
                    array([eval(x) for x in row[:n]]) for row in reader
                    ])

    # build posterior distribution
    param.update({'f':binary.PosteriorBinary(sample, param['posterior_type'])})
    out_file = param['test_folder'] + '/' + 'result.csv'

    print [(key, x) for key, x in param.iteritems() if key[:4]=='smc_']

    # setup test folder
    try: os.mkdir(param['test_folder'])
    except: pass
    if param['test_name'] is 'default': test_file = param['sys_path'] + '/src/default.py'
    else: test_file = param['test_folder'] + '.py'
    shutil.copyfile(test_file, os.path.join(param['test_folder'] , param['test_name']) + '.py')

    # setup logger
    log_stream = auxpy.logger.Logger(sys.stdout, param['test_folder'] + '/log')
    log_id = os.path.splitext(os.path.basename(log_stream.logfile.name))[0]
    sys.stdout = log_stream

    print 'start test suite of %i runs...' % param['test_runs']
    for i in xrange(param['test_runs']):

        print '\nstarting %i/%i' % (i + 1, param['test_runs'])
        result = param['test_algo'](param, verbose=verbose)
        id = ';%s;%i;%i' % (log_id, i + 1, param['f'].d)

        for j in range(5):
            try:
                file = open(out_file, 'a')
                file.write(result + id + '\n')
                file.close()
                break
            except:
                print 'Could not write to %s. Trying again in 3 seconds...'
                time.sleep(3)



def _eval_mean(param):
    '''
        Make pdf-boxplots from test files.
        
        @param param parameters
    '''

    file = open(param['test_folder'] + '/' + 'result.csv', 'r')
    reader = csv.reader(file, delimiter=';')
    X = array([
              array([eval(x) for x in row[:eval(row[-1])]]) for row in reader
        ])
    n = X.shape[0]
    d = X.shape[1]

    param['eval_title'] += '%i runs' % n

    A = zeros((5, d))
    box = param['eval_boxplot']

    X.sort(axis=0)

    for i in xrange(d):
        A[0][i] = X[:, i][0]
        for j, q in [(1, 1.0 - box), (2, 0.5), (3, box), (4, 1.0)]:
            A[j][i] = X[:, i][int(q * n) - 1] - A[:j + 1, i].sum()

    # plot with rpy
    r.pdf(paper="a4r", file=param['sys_path'] + '/' + param['test_path'] + '/' + param['test_name'] + "/eval.pdf", width=12, height=12)

    if param['eval_color']: colors = ['azure1', 'black', 'white', 'white', 'black']
    else: colors = ['grey85', 'black', 'white', 'white', 'black']

    r.par(oma=param['eval_outer_margin'], mar=param['eval_inner_margin'])

    r.barplot(A, ylim=[0, 1], names=range(0, d), axes=False, col=colors)
    r.title(main=param['eval_title'],
            line=param['eval_title_line'],
            family=param['eval_font_family'],
            cex_main=param['eval_font_cex'], font_main=1)
    r.dev_off()


if __name__ == "__main__":
    main()
