#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Execution of optimization algorithms. 

@verbatim
USAGE:
        exec <option> <file>

OPTIONS:
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

import getopt, shutil, subprocess, sys, os, time, utils
import pp
import obs

def main():

    # Parse command line options.
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hrecv', ['sa', 'ce', 'smc'])
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
    obs.read_config()
    RUN_NAME = args[0].split('.')[0]
    RUN_FOLDER = os.path.join(obs.v['RUN_PATH'], RUN_NAME)
    RUN_FILE = os.path.join(obs.v['RUN_PATH'], RUN_NAME + '.ini')
    if not os.path.isfile(RUN_FILE):
        print "The run file '%s' does not exist in the run path %s" % (RUN_NAME, RUN_FOLDER)
        sys.exit(0)
    obs.v.update({'RUN_NAME':RUN_NAME, 'RUN_FILE':RUN_FILE, 'RUN_FOLDER':RUN_FOLDER})

    # Initialize problem
    obs.read_config(os.path.join(obs.v['RUN_PATH'], RUN_NAME))

    # Process options.
    if '-c' in opts:
        try:
            for file in os.listdir(RUN_FOLDER):
                os.remove(os.path.join(RUN_FOLDER, file))
        except: pass
    if '-r' in opts:
        if '--sa' in opts: obs.v['RUN_ALGO'] = obs.sa
        if '--ce' in opts: obs.v['RUN_ALGO'] = obs.ce
        if '--smc' in opts: obs.v['RUN_ALGO'] = obs.smc
        if obs.v['RUN_ALGO'] is None: return
        else: run(v=obs.v, verbose=True)
    if '-e' in opts: plot(v=obs.v)
    if '-v' in opts:
        if not os.path.isfile(os.path.join(RUN_FOLDER, 'plot.pdf')): plot(v=obs.v)
        subprocess.Popen([obs.v['SYS_VIEWER'], os.path.join(RUN_FOLDER, 'plot.pdf')])

def run(v, verbose=False):
    """ Run algorithm from specified file and store results.
        @param v parameters
    """

    # Setup test folder.
    if not os.path.isdir(v['RUN_FOLDER']): os.mkdir(v['RUN_FOLDER'])
    shutil.copyfile(v['RUN_FILE'], os.path.join(v['RUN_FOLDER'] , v['RUN_NAME']) + '.ini')

    # Setup result file.
    result_file = v['RUN_FOLDER'] + '/' + 'result.csv'

    # Setup problem calling ubqo super constructor of algo
    v['RUN_ALGO'] = v['RUN_ALGO'](v)
    if not os.path.isfile(result_file):
        file = open(result_file, 'w')
        file.write(','.join(['OBJ', 'ALGO', 'PROBLEM', 'BEST_OBJ'] + ['S%0*d' % (3, i + 1) for i in xrange(v['RUN_ALGO'].d)] + \
                            ['TIME', 'LOG_FILE', 'LOG_NO'] + v['RUN_ALGO'].header) + '\n')
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

    # Loop over problems in testsuite.
    for problem in v['RUN_PROBLEM']:

        # Initilize solver algorithm
        v['RUN_ALGO'].__init__(v, problem)

        # Do repeated runs of the algorithm.
        if v['RUN_N'] > 1: print 'start test suite of %i runs...' % v['RUN_N']
        for i in xrange(v['RUN_N']):

            if v['RUN_N'] > 1: print '\nstarting %i/%i' % (i + 1, v['RUN_N'])
            print 'working on testsuite %s problem %d' % (v['RUN_TESTSUITE'], v['RUN_ALGO'].problem)
            result = v['RUN_ALGO'].run()

            # Write result to result.csv
            for j in xrange(4):
                try:
                    file = open(result_file, 'a')
                    file.write(','.join(['%.1f' % result['obj'], v['RUN_ALGO'].name, '%d' % v['RUN_ALGO'].problem, '%.1f' % v['RUN_ALGO'].best_obj] +
                                        ['%d' % x for x in result['soln']] +
                                        ['%.3f' % (result['time'] / 60.0)] + [log_id] + [str(i + 1)]) + '\n')
                    file.close()
                    break
                except:
                    if j < 3:
                        print 'Could not write to %s. Trying again in 3 seconds...' % result_file
                        time.sleep(3)
                    else:
                        print 'Failed to write to %s.' % result_file

def plot(v):
    # Open R-template.
    f = open(os.path.join(v['SYS_ROOT'], 'src', 'obs', 'plot.R'), 'r')
    R_script = f.read() % {'resultfile':os.path.join(v['RUN_FOLDER'], 'result.csv'),
                           'pdffile':os.path.join(v['RUN_FOLDER'], 'plot.pdf'),
                           'title':'suite: %(RUN_TESTSUITE)s %(RUN_PROBLEM)s, dim:' % v}
    f.close()

    # Copy plot.R to its run folder.
    f = open(os.path.join(v['RUN_FOLDER'], 'plot.R'), 'w')
    f.write(R_script)
    f.close()

    # Execute R-script.
    subprocess.Popen([v['SYS_R'], 'CMD', 'BATCH', '--vanilla',
                      os.path.join(v['RUN_FOLDER'], 'plot.R'),
                      os.path.join(v['RUN_FOLDER'], 'plot.Rout')]).wait()

if __name__ == "__main__":
    main()
