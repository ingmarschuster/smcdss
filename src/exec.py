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
from numpy import array

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

    # read data
    data_file = os.path.join(param['sys_path'], param['data_path'], param['data_set'], param['data_set'] + '.out')
    reader = csv.reader(open(data_file, 'r'), delimiter=',')
    data_header = reader.next()
    d = min(param['data_n_covariates'] + 1, len(data_header))
    sample = array([
                    array([eval(x) for x in row[:d]]) for row in reader
                    ])

    # build posterior distribution
    param.update({'f':binary.PosteriorBinary(sample, param['posterior_type'])})

    # setup test folder
    if not os.path.isdir(param['test_folder']): os.mkdir(param['test_folder'])
    if param['test_name'] is 'default': test_file = param['sys_path'] + '/src/default.py'
    else: test_file = param['test_folder'] + '.py'
    shutil.copyfile(test_file, os.path.join(param['test_folder'] , param['test_name']) + '.py')

    # setup result file
    result_file = param['test_folder'] + '/' + 'result.csv'
    if not os.path.isfile(result_file):
        file = open(result_file, 'w')
        file.write('\t'.join(data_header[1:d] + ['LOG_FILE', 'LOG_NO'] + param['test_algo'].header()) + '\n')
        file.close()

    # setup logger
    log_stream = auxpy.logger.Logger(sys.stdout, param['test_folder'] + '/log')
    log_id = os.path.splitext(os.path.basename(log_stream.logfile.name))[0]
    sys.stdout = log_stream

    print 'start test suite of %i runs...' % param['test_runs']
    for i in xrange(param['test_runs']):

        print '\nstarting %i/%i' % (i + 1, param['test_runs'])
        result = param['test_algo'].run(param, verbose=verbose)

        for j in range(5):
            try:
                file = open(result_file, 'a')
                file.write('\t'.join([result[0], log_id, str(i + 1) , result[1]]) + '\n')
                file.close()
                break
            except:
                print 'Could not write to %s. Trying again in 3 seconds...' % result_file
                time.sleep(3)


def _eval_mean(param):
    '''
        Make pdf-boxplots from test files.
        
        @param param parameters
    '''
    file = open(param['test_folder'] + '/' + 'result.csv', 'r')
    reader = csv.reader(file, delimiter='\t')
    data_header = reader.next()
    d = data_header.index('LOG_FILE')
    if param['eval_names']: names = data_header[:d]
    else:                   names = range(1, d + 1)

    eval = dict(TIME=[], NO_EVALS=[], LENGTH=[], NO_MOVES=[], ACC_RATE=[])
    for key in eval.keys():
        try:    eval[key].append(data_header.index(key))
        except: eval[key].append(-1)
        eval[key].append(0.0)

    X = []
    for row in reader:
        X += [array([float(x) for x in row[:d]])]
        for key in eval.keys():
            if eval[key][0] > -1: eval[key][1] += float(row[eval[key][0]])

    X = array(X)
    n = X.shape[0]
    d = X.shape[1]
    for key in eval.keys(): eval[key] = eval[key][1] / float(n)

    A = zeros((5, d))
    box = param['eval_boxplot']
    X.sort(axis=0)
    for i in xrange(d):
        A[0][i] = X[:, i][0]
        for j, q in [(1, 1.0 - box), (2, 0.5), (3, box), (4, 1.0)]:
            A[j][i] = X[:, i][int(q * n) - 1] - A[:j + 1, i].sum()

    param['eval_title'] = 'ALGO %s, DATA %s, POSTERIOR %s, DIM %i, RUNS %i, TIME %.3f, NO_EVALS %.1f' % \
        (param['test_algo'].__name__, param['data_set'], param['posterior_type'], d, n, eval['TIME'], eval['NO_EVALS'])
    if eval['LENGTH'] > 0:
        param['eval_title'] += '\nKERNEL %s, LENGTH %.1f, NO_MOVES %.1f, ACC_RATE %.3f' % \
        (param['mcmc_kernel'].__name__, eval['LENGTH'], eval['NO_MOVES'], eval['ACC_RATE'])

    # plot with rpy
    r.pdf(paper="a4r", file=param['sys_path'] + '/' + param['test_path'] + '/' + param['test_name'] + "/eval.pdf", width=12, height=12)
    r.par(oma=param['eval_outer_margin'], mar=param['eval_inner_margin'])
    r.barplot(A, ylim=[0, 1], names=names, las=2, cex_names=0.5, cex_axis=0.75, axes=True, col=param['eval_color'])
    r.title(main=param['eval_title'],
            line=param['eval_title_line'],
            family=param['eval_font_family'],
            cex_main=param['eval_font_cex'], font_main=1)
    r.dev_off()


if __name__ == "__main__":
    main()
