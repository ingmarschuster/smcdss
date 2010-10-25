'''
USAGE: saopt [OPTIONS]

    -h, --help
                                display help message
                                
    -t, --test
                                test run, do not save results in file
        
    -r, --runs       =int
                                number of tests to run
                                
    -n, --number     =int
                                number iterations
                                        
    -v, --verbose
                                turns direct output off
        
    -c, --columns    =int,int,int
                                explained column, first and last columns to cross

    -a, --absolute   =int,int,int
                                explained column, first and last columns
        
    -s, --scoretype  ={bic,hb}
                                scoretype to use: bic=Bayesian information criterion, hb=hierachical Bayes
        
    -f, --file
                                output file
    
'''

from gen import *
from sampling import *

from datetime import datetime
from copy import copy
import csv, os, glob, sys, getopt

TEST_PATH = '/home/cschafer/Documents/Python/workspace/data/testruns'

def saopt(targetDistr, n, verbose):
    '''
    Run simulated annealing optimization.
    '''
    
    start = time.time()
    indgen = binary_ind(mean='uniform', p=targetDistr.p)
    state_max = 0
    score_max = -inf
    state = indgen.rvs()
    score_state = targetDistr.lpmf(state)
    
    stdout.write("\n" + 101 * " " + "]" + "\r" + "["); progress = 0
    for t in range(1, n + 1):
        
        progress_next = 100 * t / n
        if 100 * n % t == 0:
            stdout.write((progress_next - progress) * "-")
            stdout.flush()
            progress = progress_next
        
        # generate proposal
        proposal = copy(state)
        index = random.randint(0, targetDistr.p)
        proposal[index] = proposal[index] ^ True
        score_proposal = targetDistr.lpmf(proposal)
        
        if score_max < score_proposal:
            score_max = copy(score_proposal)
            state_max = copy(proposal)
               
        if exp((score_proposal - score_state) / n * 10 * t) > random.random():
            state = proposal
            score_state = score_proposal

    state_max = where(array(state_max) == True)[0]
    
    # return results
    time_elapsed = "%.3f" % (time.time() - start)
    print "\nDone in " + time_elapsed + " seconds.\n"
    return score_max, '[' + ''.join(map(lambda x: str(x) + ', ', state_max[:-1])) + ('%i]' % state_max[-1]), time_elapsed, n

def satest(targetDistr, runs, n, filestub, test, verbose):
    '''
    Run repeated tests of sa optimization.
    '''
    # create summary and header for test report
    summary = "test runs=%i, problem size=%i" % (runs, targetDistr.p)
    summary += ", dataset=" + targetDistr.dataset + ", score type=" + targetDistr.scoretype
    
    if not test:
        if filestub == None:
            filestub = "%xsaopt" % (long(datetime.now().strftime("%m%d%H%M")))
        elif filestub.rfind("saopt") == -1:
            filestub = filestub + 'saopt'
        
        filename = "../../data/testruns/" + filestub + "_1.txt"; i = 1
        while os.path.exists(filename):
            filename = "../../data/testruns/" + filestub + ("_%i.txt" % i)
            i += 1
        file = open(filename, 'w')
        file.write(summary + "\nscore;mode;time;evals\n")
        file.close()
    
    # do test runs
    for iter in range(runs):
        # init and run simulated annealing method
        result = saopt(targetDistr=targetDistr, n=n, verbose=verbose)
        
        # write results to file
        if not test: file = open(filename, 'a')
        for item in result:
            if verbose: print str(item) + ";",
            if isinstance(item, list): item = repr(item) 
            if not test: file.write(str(item) + ";")
        if verbose: print "\n"
        if not test: file.write("\n"); file.close()

def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:n:ols:c:a:r:vf:t", \
            ['help', 'dataset=', 'eval=', 'cross=', 'absolute=', 'runs=', 'scoretype=', 'verbose', 'generator=', 'file=', 'test', 'number='])
    except getopt.error, msg:
        print msg
        sys.exit(2)
    
    #===========================================================================
    #  Default parameters
    #===========================================================================
    
    test = False
    n = 300000
    cols = [8, 9, 21]
    variates = crosscols(cols[1], cols[2])
    runs = 200
    verbose = False
    scoretype = 'hb'
    dataset = 'boston'
    filestub = None
    
    #===========================================================================
    # Process options
    #===========================================================================
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        if o in ("-r", "--runs"): runs = int(a)        
        if o in ("-c", "--cross"):
            cols = eval('[' + a + ']')
            variates = crosscols(cols[1], cols[2])
        if o in ("-a", "--absolute"):
            cols = eval('[' + a + ']')
            variates = range(cols[1], cols[2] + 1)
        if o in ("-t", "--test"): test = True  
        if o in ("-d", "--dataset"): dataset = a     
        if o in ("-n", "--number"): n = int(a)     
        if o in ("-f", "--file"): filestub = a
        if o in ("-v", "--verbose"): verbose = True
        if o in ("-s", "--scoretype"):
            scoretype = a
            if not scoretype in ('hb', 'bic'):
                print "Score type " + a + " unknown. Choose \'hb\' or \'bic\'."
                sys.exit(2)
        if o in ("-o", "-l"):
            sys.exit(0)

    targetDistr = binary_post(cols[0], variates=variates, dataset=dataset, scoretype=scoretype)
    satest(targetDistr=targetDistr, runs=runs, test=test, n=n, filestub=filestub, verbose=verbose)

if __name__ == "__main__":
    main()
