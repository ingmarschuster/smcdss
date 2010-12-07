#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date: 2010-12-03 19:08:22 +0100 (ven., 03 déc. 2010) $
    $Revision: 38 $
'''

import sys, getopt, os, time, shutil, csv
import auxpy.editcols
import binary
import default
import algos
try:
    from rpy import *
except:
    pass

def main():

    param = {}
    param.update(default.param)
    param.update({'sys_path':sys.path[0][:-4]})

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hre', ['help', 'run', 'eval'])
    except getopt.error, msg:
        print msg
        sys.exit(2)

    if len(args) == 0: sys.exit(0)

    param.update({'test_name':args[0]})
    if not os.path.isfile(param['sys_path'] + '/' + param['test_path'] + '/' + param['test_name'] + '.py'):
        print 'The test file "' + param['test_name'] + '" does not exist in the test path.'
        sys.exit(0)

    sys.path.insert(0, param['sys_path'] + '/' + param['test_path'])
    user = __import__(param['test_name'])
    param.update(user.param)

    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        if o in ("-r", "--run"):
            _testrun(param)
        if o in ("-e", "--eval"):
            _eval_mean(param)

def _testrun(param):

    if param['test_output']:
        path = param['sys_path'] + '/' + param['test_path'] + '/' + param['test_name']
        try:
            os.mkdir(path)
        except:
            pass

    algo = param['test_algo']

    data_file = auxpy.editcols.editcols(param)
    posterior_type = param['posterior_type']
    param.update({'f':binary.PosteriorBinary(data_file, posterior_type)})
    out_file = param['sys_path'] + '/' + param['test_path'] + '/' + param['test_name'] + '/' + 'result.csv'

    if param['test_name'] is 'default': test_file = param['sys_path'] + '/src/default.py'
    else: test_file = param['sys_path'] + '/' + param['test_path'] + '/' + param['test_name'] + '.py'

    shutil.copyfile(test_file, path + '/' + param['test_name'] + '.py')

    for i in range(param['test_runs']):
        result = algo(param, verbose=True)
        for j in range(5):
            try:
                file = open(out_file, 'a')
                file.write(result + '\n')
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

    file = open(param['sys_path'] + '/' + param['test_path'] + '/' + param['test_name'] + '/' + 'result.csv', 'r')
    reader = csv.reader(file, delimiter=';')
    X = array([array(eval(row[0])) for row in reader])
    n = X.shape[0]
    d = X.shape[1]

    A = zeros((5, d))
    box = param['eval_boxplot']
    X.sort(axis=0)
    for i in xrange(d):
        A[0][i] = X[:, i][0]
        for j, q in [(1, 1.0 - box), (2, 0.5), (3, box), (4, 1.0)]:
            A[j][i] = X[:, i][int(q * n) - 1] - A[:j + 1, i].sum()

    # plot with rpy
    r.pdf(paper="a4", file=param['sys_path'] + '/' + param['test_path'] + '/' + param['test_name'] + "/eval.pdf", width=12, height=12)

    if param['eval_color']: colors = ['azure1', 'black', 'white', 'white', 'black']
    else: colors = ['grey85', 'black', 'white', 'white', 'black']

    r.par(oma=param['eval_outer_margin'], mar=param['eval_inner_margin'])

    r.barplot(A, ylim=[0, 1], axes=False, col=colors)
    r.title(main=param['eval_title'],
            line=param['eval_title_line'],
            family=param['eval_font_family'],
            cex_main=param['eval_font_cex'], font_main=1)

    r.dev_off()


if __name__ == "__main__":
    main()
