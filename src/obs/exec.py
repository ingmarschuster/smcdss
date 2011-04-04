#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#    $Author: Christian Schäfer
#    $Date: 2011-03-07 17:03:12 +0100 (lun., 07 mars 2011) $
#    $Revision: 86 $

'''
USAGE:
        exec <option> <file>

OPTIONS:
        -h    display help
        -r    run optimization as specified in <file>
        -e    evaluate results obtained from running <file>
        -c    start clean run of <file>
        -v    view evaluation of <file>
'''

__version__ = "$Revision: 94 $"

import getopt, shutil, subprocess, sys, os, time, utils
import pp
import obs

def main():

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
    obs.read_config()
    RUN_NAME = os.path.splitext(os.path.basename(args[0]))[0]
    RUN_FOLDER = os.path.join(obs.v['RUN_PATH'], RUN_NAME)
    RUN_FILE = os.path.join(obs.v['RUN_PATH'], RUN_NAME + '.cfg')
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
    if '-r' in opts: run(v=obs.v, verbose=True)
    if '-e' in opts: eval(v=obs.v)
    if '-v' in opts:
        if not os.path.isfile(os.path.join(RUN_FOLDER, 'plot.pdf')): eval(v=obs.v)
        subprocess.Popen(['okular', os.path.join(RUN_FOLDER, 'plot.pdf')])

def run(v, verbose=False):
    ''' Run algorithm from specified file and store results.
        @param v parameters
    '''

    # Setup test folder.
    if not os.path.isdir(v['RUN_FOLDER']): os.mkdir(v['RUN_FOLDER'])
    shutil.copyfile(v['RUN_FILE'], os.path.join(v['RUN_FOLDER'] , v['RUN_NAME']) + '.cfg')

    # Setup result file.
    result_file = v['RUN_FOLDER'] + '/' + 'result.csv'
    if not os.path.isfile(result_file):
        file = open(result_file, 'w')
        file.write(','.join(['OBJ'] + ['S%0*d' % (3, i + 1) for i in xrange(v['RUN_ALGO'].d)] + \
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

    if v['RUN_N'] > 1: print 'start test suite of %i runs...' % v['RUN_N']
    for i in xrange(v['RUN_N']):

        if v['RUN_N'] > 1: print '\nstarting %i/%i' % (i + 1, v['RUN_N'])
        result = v['RUN_ALGO'].run()

        for j in xrange(4):
            try:
                file = open(result_file, 'a')
                file.write(','.join(['%.1f' % result['obj']] +
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
    pass

if __name__ == "__main__":
    main()
