#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Execution of optimization algorithms. 

@verbatim
USAGE:
        exec <option> <file>

OPTIONS:
        -m    start multiple instances
        -h    display help
        -r    run optimization as specified in <file>
        -e    evaluate results obtained from running <file>
        -c    start clean run of <file>
        -v    view evaluation of <file>

@endverbatim
"""

"""
@namespace obs.exec
$Author$
$Rev$
$Date$
@details
"""

import getopt
import shutil
import subprocess
import sys
import os
import time
import utils
import binary
import obs
import pp

def main():

    # Parse command line options.
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hrecvm:a:p:', ['product', 'logistic', 'gaussian'])
    except getopt.error, msg:
        print msg
        sys.exit(2)

    noargs = [o[0] for o in opts]
    if len(noargs) == 0: sys.exit(0)

    # Check for valid argument
    if len(args) == 0:
        if '-h' in opts:
            print __doc__
        else:
            print 'No file specified.'
        sys.exit(0)

    # Load run file.
    obs.read_config()
    RUN_NAME = args[0].split('.')[0]
    RUN_FOLDER = os.path.join(obs.v['RUN_PATH'], RUN_NAME)
    RUN_FILE = os.path.join(obs.v['RUN_PATH'], RUN_NAME + '.ini')
    obs.v.update({'RUN_NAME':RUN_NAME, 'RUN_FILE':RUN_FILE, 'RUN_FOLDER':RUN_FOLDER})
    if not os.path.isfile(RUN_FILE):
        print "The run file '%s' does not exist in the run path %s" % (RUN_NAME, RUN_FOLDER)
        sys.exit(0)

    # Clean up or create folder.
    if '-c' in noargs:
        if os.path.isdir(RUN_FOLDER):
            try:
                for file in os.listdir(RUN_FOLDER):
                    os.remove(os.path.join(RUN_FOLDER, file))
            except: pass
        else:
            os.mkdir(RUN_FOLDER)

    # Change run problem on command line.
    problems = None
    for o, a in opts:
        if o == '-p':
            prange = a.split(':')
            if len(prange) > 1:
                problems = range(int(prange[0]), 1 + int(prange[1]))
                if not '-m' in noargs: opts += [('-m', '1')]
            else:
                problems = [int(prange[0])]

    # Start multiple processes.
    for o, a in opts:
        if o == '-m':
            subprocesses = int(a)
            print subprocesses
            if problems is None: problems = [obs.v['RUN_PROBLEM']]
            while subprocesses > 0:
                for p in problems:
                    if os.name == 'posix':
                        subprocess.call('gnome-terminal -e "obs -p %d ' % p + 
                                        ' '.join([o + ' ' + a for (o, a) in opts if not o in ['-m', '-c', '-p']]) + ' ' + args[0] + '"', shell=True)
                    else:
                        path = os.path.join(obs.v['SYS_ROOT'], 'bin', 'obs.bat')
                        subprocess.call('start "Title" /MAX "%s" ' % path + (' -p %d ' % p) + 
                                        ' '.join([o + ' ' + a for (o, a) in opts if not o in ['-m', '-c', '-p']]) + ' ' + args[0], shell=True)
                subprocesses -= 1
            sys.exit(0)

    # Initialize problem
    obs.read_config(os.path.join(obs.v['RUN_PATH'], RUN_NAME))
    if not problems is None: obs.v['RUN_PROBLEM'] = problems[0]

    # Change model on command line.
    model = None
    if '--gaussian' in noargs: model = binary.GaussianCopulaBinary
    if '--logistic' in noargs: model = binary.LogisticBinary
    if '--product' in noargs: model = binary.ProductBinary

    # Change algorithm on command line.
    for o, a in opts:
        if o == '-a':
            if a == 'sa':
                obs.v['RUN_ALGO'] = obs.sa.sa
                obs.v['RUN_CPUS'] = None
            if a == 'scip':
                obs.v['RUN_ALGO'] = obs.scip.scip
                obs.v['RUN_CPUS'] = None
                obs.v['RUN_N'] = 1
            if a == 'ce':
                obs.v['RUN_ALGO'] = obs.ce.ce
                if not model is None:obs.v['CE_BINARY_MODEL'] = model
            if a == 'smca':
                obs.v['RUN_ALGO'] = obs.smca.smca
                if not model is None:obs.v['SMC_BINARY_MODEL'] = model
    
    # run optimization.
    if '-r' in noargs:
        if obs.v['RUN_ALGO'] is None: return
        else: run(v=obs.v, verbose=True)

    # run evaluation.
    if '-e' in noargs: plot(v=obs.v)

    # start pdf viewer after evaluation.
    if '-v' in noargs:
        if not os.path.isfile(os.path.join(RUN_FOLDER, 'plot_p%02d.pdf' % obs.v['RUN_PROBLEM'])): plot(v=obs.v)
        subprocess.Popen([obs.v['SYS_VIEWER'], os.path.join(RUN_FOLDER, 'plot_p%02d.pdf' % obs.v['RUN_PROBLEM'])])

def run(v, verbose=False):
    """
        Run algorithm from specified file and store results.
        \param v parameters
    """

    # Setup test folder.
    if not os.path.isdir(v['RUN_FOLDER']): os.mkdir(v['RUN_FOLDER'])
    shutil.copyfile(v['RUN_FILE'], os.path.join(v['RUN_FOLDER'], os.path.basename(v['RUN_FILE'])))

    # Setup result file.
    RESULT_FILE = v['RUN_FOLDER'] + '/' + 'result_p%02d.csv' % v['RUN_PROBLEM']

    # Setup problem calling ubqo super constructor of algo
    v['RUN_ALGO'] = v['RUN_ALGO'](v)
    if not os.path.isfile(RESULT_FILE):
        file = open(RESULT_FILE, 'w')
        file.write(','.join(['OBJ', 'ALGO', 'MODEL'] + ['S%0*d' % (3, i + 1) for i in xrange(v['RUN_ALGO'].d)] + \
                            ['TIME', 'LOG_FILE', 'LOG_NO']) + '\n')
        file.close()

    # Setup logger.
    if v['RUN_VERBOSE']:
        log_stream = utils.logger.Logger(sys.stdout, v['RUN_FOLDER'] + '/log')
        log_id = os.path.splitext(os.path.basename(log_stream.logfile.name))[0]
        sys.stdout = log_stream
    else:
        log_id = str(0)

    # Setup job server.
    if v['RUN_CPUS'] is None:
        v.update({'JOB_SERVER':None})
    else:
        sys.stdout.write('Starting jobserver...')
        t = time.time()
        v.update({'JOB_SERVER':pp.Server(ncpus=v['RUN_CPUS'], ppservers=())})
        print '\rjob server (%i) started in %.2f sec' % (v['JOB_SERVER'].get_ncpus(), time.time() - t)

    # Initilize solver algorithm
    v['RUN_ALGO'].__init__(v)

    # Do repeated runs of the algorithm.
    if v['RUN_N'] > 1: print 'Start test suite of %i runs...' % v['RUN_N']
    for i in xrange(v['RUN_N']):

        if v['RUN_N'] > 1: print '\nStarting %i/%i' % (i + 1, v['RUN_N'])
        print 'Working on test suite %s problem %d.' % (v['RUN_TESTSUITE'], v['RUN_ALGO'].problem)
        
        result = v['RUN_ALGO'].run()
        
        if v['RUN_ALGO'].name in ['CE', 'SMC']: model = v[v['RUN_ALGO'].name.upper() + '_BINARY_MODEL'].name
        else: model = 'none' 

        # Write result to result.csv
        for j in xrange(4):
            try:
                file = open(RESULT_FILE, 'a')
                file.write(','.join(['%.f' % result['obj'], v['RUN_ALGO'].name, model] + 
                                    ['%d' % x for x in result['soln']] + 
                                    ['%.3f' % (result['time'] / 60.0), log_id, str(i + 1) + '\n']))
                file.close()
                break
            except:
                if j < 3:
                    print 'Could not write to %s. Trying again in 3 seconds...' % RESULT_FILE
                    time.sleep(3)
                else:
                    print 'Failed to write to %s.' % RESULT_FILE

def plot(v):
    
    # Open R-template.
    title = 'suite: %(RUN_TESTSUITE)s %(RUN_PROBLEM)s, dim:' % v
    if not v['EVAL_TITLE']: title = ''
    colors = ', '.join(["'" + str(x) + "'" for x in v['EVAL_COLOR']])
    mar = ', '.join([str(x) for x in v['EVAL_INNER_MARGIN']])
    
    testsuite, problem = v['RUN_TESTSUITE'], v['RUN_PROBLEM']

    file = open(os.path.join(obs.v['DATA_PATH'], testsuite, testsuite + '_%02d.dat' % problem), 'r')
    primal_bound = eval(file.readline())
    file.close()

    f = open(os.path.join(v['SYS_ROOT'], 'src', 'obs', 'template_plot.R'), 'r')
    R_script = f.read() % {'resultfile':os.path.join(v['RUN_FOLDER'], 'result_p%02d.csv' % problem).replace('\\', '\\\\'),
                           'pdffile':os.path.join(v['RUN_FOLDER'], 'plot_p%02d.pdf' % problem).replace('\\', '\\\\'),
                           'title':title, 'colors':colors, 'mar':mar, 'type':v['EVAL_TYPE'],
                           'bars':v['EVAL_BARS'], 'exact':v['EVAL_EXACT'], 'n':v['EVAL_N'], 'primal_bound':primal_bound}
    f.close()

    # Copy plot.R to its run folder.
    f = open(os.path.join(v['RUN_FOLDER'], 'plot_p%02d.R' % problem), 'w')
    f.write(R_script)
    f.close()

    # Execute R-script.
    subprocess.Popen([v['SYS_R'], 'CMD', 'BATCH', '--vanilla',
                      os.path.join(v['RUN_FOLDER'], 'plot_p%02d.R' % problem),
                      os.path.join(v['RUN_FOLDER'], 'plot_p%02d.Rout' % problem)]).wait()

if __name__ == "__main__":
    main()
