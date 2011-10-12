#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Execution of integration algorithms. 

@verbatim
USAGE:
        exec [option] [file]

OPTIONS:
        -m    start multiple instances
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

from binary import *
import getopt
import shutil
import csv
import datetime
import subprocess
import os
import sys
import ibs
import pp
import smc
import mcmc

def main():
    """ Main method. """

    # Parse command line options.
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hrecvm:a:')
    except getopt.error, msg:
        print msg
        sys.exit(2)

    noargs = [o[0] for o in opts]
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
    RUN_FILE = os.path.join(ibs.v['RUN_PATH'], RUN_NAME)
    
    # Check for algorithm options.
    for o, a in opts:
        if o == '-a':
            if not a in ['smc', 'mcmc', 'amcmc']:
                print 'Algorithm %s not recognized.' % a
                sys.exit(0)
            if a == 'smc':
                ibs.v['RUN_ALGO'] = smc.smc
            if a == 'mcmc':
                ibs.v['RUN_ALGO'] = mcmc.mcmc
                ibs.v['MCMC_KERNEL'] = mcmc.SymmetricMetropolisHastings
            if a == 'amcmc':
                ibs.v['RUN_ALGO'] = mcmc.mcmc
                ibs.v['MCMC_KERNEL'] = mcmc.AdaptiveMetropolisHastings
            RUN_NAME += '_' + a
    RUN_FOLDER = os.path.join(ibs.v['RUN_PATH'], RUN_NAME)
    EVAL_FILE = os.path.join(RUN_FOLDER, RUN_NAME)
    ibs.v.update({'RUN_NAME':RUN_NAME, 'RUN_FILE':RUN_FILE + '.ini', 'EVAL_FILE':EVAL_FILE + '.ini', 'RUN_FOLDER':RUN_FOLDER})    
    
    # Clean up or create folder.
    if '-c' in noargs:
        if os.path.isdir(RUN_FOLDER):
            try:
                for file in os.listdir(RUN_FOLDER):
                    os.remove(os.path.join(RUN_FOLDER, file))
            except: pass
        else:
            os.mkdir(RUN_FOLDER)
            
    # Start multiple processes.
    for o, a in opts:
        if o == '-m':
            k = int(a)
            while k > 0:
                if os.name == 'posix':
                    subprocess.call('gnome-terminal -e "ibs ' + ' '.join([o + ' ' + a for (o, a) in opts if not o in ['-m', '-c']]) + ' ' + args[0] + '"', shell=True)
                else:
                    path = os.path.abspath(os.path.join(os.path.join(*([os.getcwd()] + ['..']*1)), 'bin', 'ibs.bat'))
                    subprocess.call('start "ibs" /MAX "%s" ' % path + 
                                    ' '.join([o + ' ' + a for (o, a) in opts if not o in ['-m', '-c']]) + ' ' + args[0], shell=True)
                k -= 1
            sys.exit(0)
   
    # Process options.
    if '-r' in noargs:
        if not (os.path.isfile(RUN_FILE + '.ini')):
            print "The run file '%s' does not exist in the path %s" % (os.path.basename(RUN_FILE), os.path.dirname(RUN_FILE))
            sys.exit(0)
        ibs.read_config(RUN_FILE)
        run(v=ibs.v)
        
    if '-e' in noargs:
        if not (os.path.isfile(EVAL_FILE + '.ini')):
            print "The evaluation file '%s' does not exist in the path %s" % (os.path.basename(EVAL_FILE), os.path.dirname(EVAL_FILE))
            sys.exit(0)
        ibs.read_config(EVAL_FILE)
        plot(v=ibs.v)
    
    if '-v' in noargs:
        if not os.path.isfile(os.path.join(RUN_FOLDER, 'plot.pdf')): plot(v=ibs.v)
        subprocess.Popen([ibs.v['SYS_VIEWER'], os.path.join(RUN_FOLDER, 'plot.pdf')])

def run(v):
    """ 
        Run algorithm from specified file and store results.
        @param v parameters
    """

    # read data or group files
    if v['POSTERIOR_TYPE'] == 're': v = readGroups(v)
    else: v = readData(v)

    # Setup test folder.
    if not os.path.isdir(v['RUN_FOLDER']): os.mkdir(v['RUN_FOLDER'])
    if not os.path.isfile(v['EVAL_FILE']): shutil.copyfile(v['RUN_FILE'], v['EVAL_FILE'])

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
        sys.stdout.write('Starting jobserver...')
        t = time.time()
        v.update({'JOB_SERVER':pp.Server(ncpus=v['RUN_CPUS'], ppservers=())})
        print '\rJob server (%i) started in %.2f sec' % (v['JOB_SERVER'].get_ncpus(), time.time() - t)

    if v['RUN_N'] > 1: print 'Start test suite of %i runs...' % v['RUN_N']
    for i in xrange(v['RUN_N']):

        if v['RUN_N'] > 1: print '\nStarting %i/%i' % (i + 1, v['RUN_N'])
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
    DATA_FILE = os.path.join(v['SYS_ROOT'], v['DATA_PATH'], v['DATA_DATA_FILE'])
    if not DATA_FILE[-4:].lower() == '.csv': DATA_FILE += '.csv'
    reader = csv.reader(open(DATA_FILE, 'rU'), delimiter=',')
    DATA_HEADER = reader.next()
    d = len(DATA_HEADER)
    if not isinstance(v['DATA_EXPLAINED'], str): Y_pos = v['DATA_EXPLAINED'] - 1
    else: Y_pos = DATA_HEADER.index(v['DATA_EXPLAINED']) 
    
    # add constant to data header. add constant to the covariates if it is not in the principal components.
    if v['DATA_CONST']:
        DATA_HEADER += ['CONST']
        if v['DATA_PCA'] is None or not 'CONST' in v['DATA_PCA']:
            v['DATA_COVARIATES'] = 'CONST+' + v['DATA_COVARIATES']

    # read covariate positions
    cindex = list()
    if not v['DATA_COVARIATES'] is None:
        for covrange in str(v['DATA_COVARIATES']).split('+'):
            covrange = covrange.split(':')
            if len(covrange) == 1:covrange += [covrange[0]]
            for i in xrange(2):
                if covrange[i].isdigit(): covrange[i] = int(covrange[i]) - 1
                elif covrange[i] == 'inf': covrange[i] = d - 1
                else: covrange[i] = DATA_HEADER.index(covrange[i])
            cindex += range(covrange[0], covrange[1] + 1)

    # read principal components positions
    pindex = list()
    if not v['DATA_PCA'] is None:
        for pcarange in str(v['DATA_PCA']).split('+'):
            pcarange = pcarange.split(':')
            if len(pcarange) == 1:pcarange += [pcarange[0]]
            for i in xrange(2):
                if pcarange[i].isdigit(): pcarange[i] = int(pcarange[i]) - 1
                elif pcarange[i] == 'inf': pcarange[i] = d - 1
                else: pcarange[i] = DATA_HEADER.index(pcarange[i])
            pindex += range(pcarange[0], pcarange[1] + 1)
    v['DATA_PCA'] = len(pindex)
    
    sample = list()
    for row in reader:
        if len(row) > 0 and not row[Y_pos] == 'NA':
            row += [c for c in ['1'] if v['DATA_CONST']] # constant column
            sample += [numpy.array([
                eval(x) for x in
                    [row[Y_pos]] + # observation column
                    [row[i] for i in pindex] + # principal component colums
                    [row[i] for i in cindex] # covariate columns
                ])]
    sample = numpy.array(sample)
    
    # use just the first DATA_MAX_OBS observations
    v['DATA_MAX_OBS'] = min(v['DATA_MAX_OBS'], sample.shape[0])
    sample = sample[:v['DATA_MAX_OBS'], :]

    PCA_HEADER = [DATA_HEADER[i] for i in pindex]
    DATA_HEADER = [DATA_HEADER[i] for i in cindex]

    # for each interaction column store the columns of the two main effects
    if v['DATA_MAIN_EFFECTS']:
        INTERACTIONS = numpy.array([[DATA_HEADER.index(icol[0]), DATA_HEADER.index(icol[1]), DATA_HEADER.index(icol[2])]
                        for icol in [[col] + col.split('.x.') for col in DATA_HEADER if '.x.' in col] if not icol[1] == icol[2]])
    else:
        INTERACTIONS = numpy.array([])

    v.update({'DATA_HEADER' : DATA_HEADER, 'INTERACTIONS' : INTERACTIONS, 'PCA_HEADER' : PCA_HEADER})
    v.update({'f': PosteriorBinary(Y=sample[:, 0], X=sample[:, 1:], param=v)})
    return v


def readGroups(v):
    """ 
        Reads the data file and a group file to set up a random effect model.
        @param v parameters
    """
    # open the data file to load the marker positions
    DATA_FILE = os.path.join(v['SYS_ROOT'], v['DATA_PATH'], v['DATA_DATA_FILE'])
    if not DATA_FILE[-4:].lower() == '.csv': DATA_FILE += '.csv'
    dreader = csv.reader(open(DATA_FILE, 'rU'), delimiter=',')
    DATA_HEADER = dreader.next()

    # open the group file to load all group information
    GROUP_FILE = os.path.join(v['SYS_ROOT'], v['DATA_PATH'], v['DATA_GROUP_FILE'])
    if not GROUP_FILE[-4:].lower() == '.csv': GROUP_FILE += '.csv'
    greader = csv.reader(open(GROUP_FILE, 'rU'), delimiter=',')
    greader.next()
    GROUPS_HEADER = list()
    GROUPS_ALL = dict()
    for g in greader:
        if not len(g) > 0: continue
        GROUPS_HEADER += [g[0]]
        GROUPS_ALL.update({g[0]:{'start':g[1], 'end':g[2]}})

    # pick groups from setup file
    gindex = list()
    for grange in v['DATA_GROUPS'].split('+'):
        grange = grange.split(':')
        if len(grange) == 1:grange += [grange[0]]
        for i in xrange(2):
            if grange[i].isdigit(): grange[i] = int(grange[i]) - 1
            elif grange[i] == 'inf': grange[i] = len(GROUPS_HEADER) - 1
            else: grange[i] = GROUPS_HEADER.index(grange[i])
        gindex += range(grange[0], grange[1] + 1)

    GROUPS_HEADER = [GROUPS_HEADER[i] for i in gindex]
    GROUPS = list()
    for group in GROUPS_HEADER:
        try:
            GROUPS += [{'start':DATA_HEADER.index(GROUPS_ALL[group]['start']), 'end':DATA_HEADER.index(GROUPS_ALL[group]['end'])}]
        except:
            print "Covariate %s in group %s was not found. Aborted." % (GROUPS_ALL[group]['start'], group)
            sys.exit(0)

    # pick data for the groups
    cindex = list()
    for group in GROUPS: cindex += range(group['start'], group['end'] + 1)
    if not isinstance(v['DATA_EXPLAINED'], str): Y_pos = v['DATA_EXPLAINED'] - 1
    else: Y_pos = DATA_HEADER.index(v['DATA_EXPLAINED'])

    sample = numpy.array([numpy.array([eval(x) for x in [row[Y_pos]] + [row[i] for i in cindex]])
                          for row in dreader if len(row) > 0 and not row[Y_pos] == 'NA'])

    # use just the first DATA_MAX_OBS observations
    v['DATA_MAX_OBS'] = min(v['DATA_MAX_OBS'], sample.shape[0])
    sample = sample[:v['DATA_MAX_OBS'], :]

    # initialize posterior
    v.update({'GROUPS':GROUPS, 'DATA_HEADER' : GROUPS_HEADER})
    v.update({'f': PosteriorBinary(Y=sample[:, 0], X=sample[:, 1:], param=v)})
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

    # Determine algorithm (might have been set on command line)
    algo = v['RUN_ALGO'].__name__
    if v['EVAL_FILE'][-8:] == 'mcmc.ini': algo = 'mcmc'

    # Format title.   
    title = 'ALGO %s, DATA %s, POSTERIOR %s, DIM %i, RUNS %i, TIME %s, NO_EVALS %.1f' % \
            (algo, v['DATA_DATA_FILE'], v['POSTERIOR_TYPE'], d, n, v['TIME'], v['NO_EVALS'])
    if eval['LENGTH'] > 0:
        title += '\nKERNEL %s, LENGTH %.1f, NO_MOVES %.1f, ACC_RATE %.3f' % \
            (v['MCMC_KERNEL'].__name__, v['LENGTH'], v['NO_MOVES'], v['ACC_RATE'])
    if verbose: print title + '\n'

    # Format dictionary.
    v.update({'EVAL_BOXPLOT':', '.join(['%.6f' % x for x in numpy.reshape(A, (5 * d,))]),
              'EVAL_DIM':str(d), 'EVAL_XAXS':A.shape[1] * 1.2 + 1,
              'EVAL_PDF':os.path.join(v['RUN_FOLDER'], 'plot.pdf').replace('\\', '/'),
              'EVAL_TITLE_TEXT':title})
    for key in ['EVAL_OUTER_MARGIN', 'EVAL_INNER_MARGIN']: v[key] = ', '.join([str(x) for x in v[key]])
    v['EVAL_NAMES'] = ', '.join(["'" + str(x) + "'" for x in v['EVAL_NAMES']])

    # Create R-script.
    if v['EVAL_LINES'] > 1:
        R = """#\n# This file was automatically generated.\n#\n
        # Boxplot data from repeated runs
        boxplot = t(array(c(%(EVAL_BOXPLOT)s),c(%(EVAL_DIM)s,5)))\n
        # Covariate names
        names = c(%(EVAL_NAMES)s)\n
        k=%(EVAL_LINES)s
        d=length(names)
        l=ceiling(d/k)
        # Create PDF-file
        pdf(paper='a4', file='%(EVAL_PDF)s', height=20, width=20)
        par(mfrow=c(k,1), oma=c(%(EVAL_OUTER_MARGIN)s), mar=c(%(EVAL_INNER_MARGIN)s))
        for(i in 1:k) {
          start=(i-1)*l+1
          end=min(i*l,d)
          barplot(boxplot[,start:end], ylim=c(0, 1), names=names[start:end], las=2, cex.names=0.5, cex.axis=0.75, axes=TRUE, col=c('%(EVAL_COLOR)s','black','white','white','black'), xaxs='i')
        }
        """ % v
    else:
        R = """#\n# This file was automatically generated.\n#\n
        # Boxplot data from repeated runs
        boxplot = t(array(c(%(EVAL_BOXPLOT)s),c(%(EVAL_DIM)s,5)))\n
        # Covariate names
        names = c(%(EVAL_NAMES)s)\n
        # Create PDF-file
        pdf(file='%(EVAL_PDF)s', height=%(EVAL_HEIGHT)s, width=%(EVAL_WIDTH)s)
        par(oma=c(%(EVAL_OUTER_MARGIN)s), mar=c(%(EVAL_INNER_MARGIN)s))
        barplot(boxplot, ylim=c(0, 1), names=names, las=2, cex.names=0.5, cex.axis=0.75, axes=TRUE, col=c('%(EVAL_COLOR)s','black','white','white','black'), xaxs='i', xlim=c(-1, %(EVAL_XAXS)s))
        """ % v
    if v['EVAL_TITLE']:
        if v['EVAL_LINES'] > 1:
            R += """
            mtext('%(EVAL_TITLE_TEXT)s', family='%(EVAL_FONT_FAMILY)s', line=-%(EVAL_TITLE_LINE)s, cex.main=%(EVAL_FONT_CEX)s, outer=TRUE)
            """ % v
        else:
            R += """
            title(main='%(EVAL_TITLE_TEXT)s', line=%(EVAL_TITLE_LINE)s, family='%(EVAL_FONT_FAMILY)s', cex.main=%(EVAL_FONT_CEX)s, font.main=1)
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
