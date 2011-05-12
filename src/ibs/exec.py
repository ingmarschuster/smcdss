#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Execution of integration algorithms. 

@verbatim
USAGE:
        exec [option] [file]

OPTIONS:
        -h    display help
        -r    run integration of posterior as specified in [file]
        -e    evaluate results obtained from running [file]
        -c    start clean run of [file]
        -v    view evaluation of [file]

@endverbatim
"""

"""
@namespace ibs.exec
$Author$
$Rev$
$Date$
@details
"""

import getopt, shutil, csv, datetime, subprocess, os, sys
import pp
import ibs
from binary import *

def main():
    """ Main method. """

    # Parse command line options.
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hrecv')
    except getopt.error, msg:
        print msg
        sys.exit(2)

    opts = [o[0] for o in opts]
    if len(opts) == 0: sys.exit(0)
    if len(args) == 0:
        if '-h' in opts:
            print __doc__
        else:
            print 'No file specified.'
        sys.exit(0)

    # Load run file.
    ibs.read_config()
    RUN_NAME = os.path.splitext(os.path.basename(args[0]))[0]
    RUN_FOLDER = os.path.join(ibs.v['RUN_PATH'], RUN_NAME)
    RUN_FILE = os.path.join(ibs.v['RUN_PATH'], RUN_NAME + '.ini')
    if not os.path.isfile(RUN_FILE):
        print "The run file '%s' does not exist in the run path %s" % (RUN_NAME, RUN_FOLDER)
        sys.exit(0)
    ibs.v.update({'RUN_NAME':RUN_NAME, 'RUN_FILE':RUN_FILE, 'RUN_FOLDER':RUN_FOLDER})

    # Initialize problem
    ibs.read_config(os.path.join(ibs.v['RUN_PATH'], RUN_NAME))

    # Process options.
    if '-c' in opts:
        try: shutil.rmtree(RUN_FOLDER)
        except: pass
    if '-r' in opts: run(v=ibs.v)
    if '-e' in opts: plot(v=ibs.v)
    if '-v' in opts:
        if not os.path.isfile(os.path.join(RUN_FOLDER, 'plot.pdf')): plot(v=ibs.v)
        subprocess.Popen(['okular', os.path.join(RUN_FOLDER, 'plot.pdf')])

def run(v):
    """ 
        Run algorithm from specified file and store results.
        @param v parameters
    """

    v = readData(v)
    # Setup test folder.
    if not os.path.isdir(v['RUN_FOLDER']): os.mkdir(v['RUN_FOLDER'])
    shutil.copyfile(v['RUN_FILE'], os.path.join(v['RUN_FOLDER'] , v['RUN_NAME']) + '.py')

    # Setup result file.
    result_file = v['RUN_FOLDER'] + '/' + 'result.csv'
    if not os.path.isfile(result_file):
        file = open(result_file, 'w')
        file.write(','.join(v['DATA_HEADER'] + ['LOG_FILE', 'LOG_NO'] + v['RUN_ALGO'].header) + '\n')
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
        sys.stdout.write('starting jobserver...')
        t = time.time()
        v.update({'JOB_SERVER':pp.Server(ncpus=v['RUN_CPUS'], ppservers=())})
        print '\rjob server (%i) started in %.2f sec' % (v['JOB_SERVER'].get_ncpus(), time.time() - t)

    if v['RUN_N'] > 1: print 'start test suite of %i runs...' % v['RUN_N']
    for i in xrange(v['RUN_N']):

        if v['RUN_N'] > 1: print '\nstarting %i/%i' % (i + 1, v['RUN_N'])
        result = v['RUN_ALGO'].run(v)

        for j in xrange(4):
            try:
                file = open(result_file, 'a')
                file.write(','.join([result[0], log_id, str(i + 1) , result[1]]) + '\n')
                file.close()

                if len(result) > 2:
                    for file_name, i in [('pd', 2), ('ar', 3)]:
                        file = open(v['RUN_FOLDER'] + '/' + '%s.csv' % file_name, 'a')
                        file.write(result[i] + '\n')
                        file.close()
                break
            except:
                if j < 3:
                    print 'Could not write to %s. Trying again in 3 seconds...' % result_file
                    time.sleep(3)
                else:
                    print 'Failed to write to %s.' % result_file

def readData(v):
    """ 
        Reads the data file and adds the posterior distribution to the
        parameters.
        @param v parameters
    """
    DATA_FILE = os.path.join(v['SYS_ROOT'], v['DATA_PATH'], v['DATA_SET'], v['DATA_SET'] + '.csv')
    reader = csv.reader(open(DATA_FILE, 'r'), delimiter=',')
    DATA_HEADER = reader.next()
    d = len(DATA_HEADER)
    if not isinstance(v['DATA_EXPLAINED'], str):Y_pos = v['DATA_EXPLAINED'] - 1
    else: Y_pos = DATA_HEADER.index(v['DATA_EXPLAINED'])
    if not isinstance(v['DATA_FIRST_COVARIATE'], str):X_first = v['DATA_FIRST_COVARIATE'] - 1
    else: X_first = DATA_HEADER.index(v['DATA_FIRST_COVARIATE'])
    if not isinstance(v['DATA_LAST_COVARIATE'], str): X_last = min(v['DATA_LAST_COVARIATE'] - 1, d)
    else: X_last = DATA_HEADER.index(v['DATA_LAST_COVARIATE'])
    sample = numpy.array([numpy.array([eval(x) for x in [row[Y_pos]] + row[X_first:X_last]])
                          for row in reader if len(row) > 0 and not row[Y_pos] == 'NA'])
    v.update({'f': PosteriorBinary(Y=sample[:, 0], X=sample[:, 1:], param=v),
              'DATA_HEADER' : DATA_HEADER[X_first:X_last]})
    return v

def plot(v, verbose=True):
    """ 
        Create pdf-boxplots from run files.
        @param v parameters
    """

    if not os.path.isfile(os.path.join(v['RUN_FOLDER'], 'result.csv')):
        print 'No file %s found.' % os.path.join(v['RUN_FOLDER'], 'result.csv')
        sys.exit(2)
    file = open(os.path.join(v['RUN_FOLDER'], 'result.csv'), 'r')
    reader = csv.reader(file, delimiter=',')

    # Read header names.
    data_header = reader.next()
    d = data_header.index('LOG_FILE')
    if v['EVAL_NAMES']: v['EVAL_NAMES'] = data_header[:d]
    else:               v['EVAL_NAMES'] = range(1, d + 1)

    # Prepare header.
    eval = dict(TIME=[], NO_EVALS=[], LENGTH=[], NO_MOVES=[], ACC_RATE=[])
    for key in eval.keys():
        try:    eval[key].append(data_header.index(key))
        except: eval[key].append(-1)
        eval[key].append(0.0)

    # Read data.
    X = list()
    for i, row in enumerate(reader):
        if i == v['EVAL_MAX_DATA']: break
        X += [numpy.array([float(x) for x in row[:d]])]
        for key in eval.keys():
            if eval[key][0] > -1: eval[key][1] += float(row[eval[key][0]])
    file.close()
    X = numpy.array(X)

    # Compute averages.
    n = X.shape[0]
    d = X.shape[1]
    for key in eval.keys(): eval[key] = eval[key][1] / float(n)
    eval['TIME'] = str(datetime.timedelta(seconds=int(eval['TIME'])))
    v.update(eval)

    # Compute quantiles for the box plot.
    A = numpy.zeros((5, d))
    box = v['EVAL_BOXPLOT']
    X.sort(axis=0)
    for i in xrange(d):
        A[0][i] = X[:, i][0]
        for j, q in [(1, 1.0 - box), (2, 0.5), (3, box), (4, 1.0)]:
            A[j][i] = X[:, i][int(q * n) - 1] - A[:j + 1, i].sum()

    # Format title.
    title = 'ALGO %s, DATA %s, POSTERIOR %s, DIM %i, RUNS %i, TIME %s, NO_EVALS %.1f' % \
            (v['RUN_ALGO'].__name__, v['DATA_SET'], v['POSTERIOR_TYPE'], d, n, v['TIME'], v['NO_EVALS'])
    if eval['LENGTH'] > 0:
        title += '\nKERNEL %s, LENGTH %.1f, NO_MOVES %.1f, ACC_RATE %.3f' % \
            (v['MCMC_KERNEL'].__name__, v['LENGTH'], v['NO_MOVES'], v['ACC_RATE'])
    if verbose: print title + '\n'

    # Format dictionary.
    v.update({'EVAL_BOXPLOT':', '.join(['%.6f' % x for x in numpy.reshape(A, (5 * d,))]),
              'EVAL_DIM':str(d), 'EVAL_XAXS':A.shape[1] * 1.2 + 1,
              'EVAL_PDF':os.path.join(v['RUN_FOLDER'], 'plot.pdf'),
              'EVAL_TITLE':title})
    for key in ['EVAL_OUTER_MARGIN', 'EVAL_INNER_MARGIN']: v[key] = ', '.join([str(x) for x in v[key]])
    for key in ['EVAL_NAMES', 'EVAL_COLOR']: v[key] = ', '.join(["'" + str(x) + "'" for x in v[key]])

    # Create R-script.

    R = """#\n# This file was automatically generated.\n#\n
    # Boxplot data from repeated runs
    boxplot = t(array(c(%(EVAL_BOXPLOT)s),c(%(EVAL_DIM)s,5)))\n
    # Covariate names
    names = c(%(EVAL_NAMES)s)\n
    # Create PDF-file
    pdf(file='%(EVAL_PDF)s', height=%(EVAL_HEIGHT)s, width=%(EVAL_WIDTH)s)
    par(oma=c(%(EVAL_OUTER_MARGIN)s), mar=c(%(EVAL_INNER_MARGIN)s))
    barplot(boxplot, ylim=c(0, 1), names=names, las=2, cex.names=0.5, cex.axis=0.75, axes=TRUE, col=c(%(EVAL_COLOR)s), xaxs='i', xlim=c(-1, %(EVAL_XAXS)s))
    """ % v
    if v['EVAL_TITLE']: R += """
    title(main='%(EVAL_TITLE)s', line=%(EVAL_TITLE_LINE)s, family='%(EVAL_FONT_FAMILY)s', cex.main=%(EVAL_FONT_CEX)s, font.main=1)
    """ % v
    R += 'dev.off()'
    R = R.replace('    ', '')
    R_file = open(os.path.join(v['SYS_ROOT'], v['RUN_PATH'], v['RUN_NAME'], 'plot.R'), 'w')
    R_file.write(R)
    R_file.close()

    # Execute R-script.
    subprocess.Popen([v['SYS_R'], 'CMD', 'BATCH', '--vanilla',
                      os.path.join(v['RUN_FOLDER'], 'plot.R'),
                      os.path.join(v['RUN_FOLDER'], 'plot.Rout')]).wait()

if __name__ == "__main__":
    main()
