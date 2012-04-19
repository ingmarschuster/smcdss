#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Computes or plots the posterior mean of a Bayesian variable selection problem.
@verbatim
USAGE:
        ibs [option] [file]

OPTIONS:
        -r    run Monte Carlo algorithm as specified [file]
        -c    start clean run of [file]
        -e    evaluate results of [file]
        -v    open plot of [file] with standard viewer
        -m    start multiple processes
@endverbatim
"""

"""
@namespace gui.shell
"""

import algo.smc
import config
import getopt
import os
import parallel.pp as pp
import plot
import shutil
import subprocess
import sys
import time
import utils.logger

def main():
    """ Main method. """

    # Parse command line options.
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'recvm:a:')
    except getopt.error, msg:
        print msg
        sys.exit(2)

    # Check arguments and options.
    noargs = [o[0] for o in opts]
    if len(opts) == 0:
        print __doc__.replace('@verbatim', '').replace('@endverbatim', '')
        sys.exit(0)
    if len(args) == 0:
        print 'No file specified.'
        sys.exit(0)

    # Determine full file name.
    default = config.import_config(config.get_default_filename())
    if not args[0].endswith('.ini'): args[0] += '.ini'
    myconfig = config.import_config(os.path.join(default['path/run'], args[0]))

    # Clean up or create folder.
    if '-c' in noargs:
        clean_folder(myconfig)

    # Start multiple processes.
    for o, a in opts:
        if o == '-m':
            k = int(a)
            while k > 0:
                if os.name == 'posix':
                    subprocess.call('gnome-terminal -e "ibs ' + ' '.join([o + ' ' + a for (o, a) in opts if not o in ['-m', '-c']]) + ' ' + args[0] + '"', shell=True)
                else:
                    path = os.path.abspath(os.path.join(os.path.join(*([os.getcwd()] + ['..'] * 1)), 'bin', 'ibs.bat'))
                    subprocess.call('start "ibs" /MAX "%s" ' % path +
                                    ' '.join([o + ' ' + a for (o, a) in opts if not o in ['-m', '-c']]) + ' ' + args[0], shell=True)
                k -= 1
            sys.exit(0)

    # Process options.
    if '-r' in noargs:
        if not (os.path.isfile(myconfig['run/file'])):
            print "The run file '%s' does not exist in the path %s" % \
                (os.path.basename(myconfig['run/file']), os.path.dirname(myconfig['run/file']))
            sys.exit(0)
        run(myconfig)

    #  Create PDF file.
    if '-e' in noargs:
        create_pdf(myconfig)

    # Open PDF viewer.
    if '-v' in noargs:
        show_pdf(myconfig)


def create_pdf(config):
    '''
        Create PDF file.
    '''
    if not (os.path.isfile(config['eval/file'])):
        sys.stdout.write("\rThe evaluation file '%s' does not exist in the path %s\n" % \
            (os.path.basename(config['eval/file']), os.path.dirname(config['eval/file'])))
        return False
    return plot.plot_R(config)

def show_pdf(config):
    '''
        Open PDF viewer.
    '''
    if not create_pdf(config) is None: return
    if config['run/verbose']:
        sys.stdout.write('\rOpening %s\n' % os.path.join(config['run/folder'], 'plot.pdf'))
    subprocess.Popen([config['path/viewer'], os.path.join(config['run/folder'], 'plot.pdf')])

def clean_folder(config):
    '''
        Clean up or create folder.
    '''
    if os.path.isdir(config['run/folder']):
        try:
            for filename in os.listdir(config['run/folder']):
                os.remove(os.path.join(config['run/folder'], filename))
        except: pass
    else:
        os.mkdir(config['run/folder'])

def prepare_run(config):
    '''
        Prepare folder and logger for run. 
    '''
    # Setup test folder.
    if not os.path.isdir(config['run/folder']):
        os.mkdir(config['run/folder'])
    if not os.path.isfile(config['eval/file']):
        shutil.copyfile(config['run/file'], config['eval/file'])

    # Setup result file.
    config['result_file'] = config['run/folder'] + '/' + 'result.csv'
    if not os.path.isfile(config['result_file']):
        f = open(config['result_file'], 'w')
        f.write(','.join(config['data/static_header'] + config['data/free_header']
                         + ['LOG_FILE', 'LOG_NO'] + [config['run/algo']]) + '\n')
        f.close()

    return config

def prepare_logger(config):
    '''
        Setup logger.
    '''
    if config['run/verbose']:
        log_stream = utils.logger.Logger(sys.stdout, config['run/folder'] + '/log')
        config['log_id'] = os.path.splitext(os.path.basename(log_stream.logfile.name))[0]
        sys.stdout = log_stream
    else:
        config['log_id'] = str(0)
    return config

def prepare_job_server(config):
    '''
        Setup job server.
    '''
    job_server = None
    if config['run/cpus'] > 1:
        t = time.time()
        sys.stdout.write('Starting job server...')
        job_server = pp.Server(ncpus=config['run/cpus'], ppservers=())
        sys.stdout.write('\rJob server (%i) started in %.2f sec' % (job_server.get_ncpus(), time.time() - t))

    return job_server

def write_result_file(result, index, config):
    '''
        Write results to file.
    '''
    try:
        f = open(config['result_file'], 'a')
        f.write(','.join([result[0], config['log_id'], str(index + 1) , result[1]]) + '\n')
        f.close()

        for filename, i in [('pd', 2), ('ar', 3)]:
            f = open(config['run/folder'] + '/' + '%s.csv' % filename, 'a')
            f.write(result[i] + '\n')
            f.close()
    except:
        sys.stdout.write('\rFailed to write to %s.' % config['result_file'])

def run(myconfig):
    """ 
        Run algorithm from specified file and store results.
        \param v parameters
    """

    # Read data.
    myconfig = config.import_data(myconfig)

    # Start SMC.
    for index in xrange(myconfig['run/n']):

        if myconfig['run/n'] > 1:
            print '\nStarting %i/%i' % (index + 1, myconfig['run/n'])

        job_server = prepare_job_server(myconfig)
        myconfig = prepare_logger(myconfig)
        myconfig = prepare_run(myconfig)

        smc = algo.smc.AnnealedSMC(myconfig, target=1.0, job_server=job_server)
        smc.sample()
        write_result_file(smc.get_csv(), index, myconfig)


if __name__ == "__main__":
    main()
