'''
USAGE: ceopt [OPTIONS]

    -h, --help
                                display help message
                                
    -t, --test
                                test run, do not save results in file
                                
    -e, --eval       =file.txt
                                evaluate result file, create file.eval.txt and file.pdf 
        
    -n, --nparticles =int
                                numer of particles
        
    -r, --runs       =int
                                number of tests to run
                                
    -p, --plot
                                plot steps for one run
        
    -v, --verbose
                                turns direct output off
        
    -c, --columns    =int,int,int
                                explained column, first and last columns to cross
        
    -s, --scoretype  ={bic,hb}
                                scoretype to use: bic=Bayesian information criterion, hb=hierachical Bayes
        
    -w, --weighted
                                enable weigted averages and regressions
        
    -f, --file
                                output file
        
    -g, --generator  ={ilm}
                                generators to use: i=independent, m=multinormal, l=logistic regression
        
    -1 =float[,float,float]     parameters to pass to first generator
        
    -2 =float[,float,float]     parameters to pass to second generator
        
    -3 =float[,float,float]     parameters to pass to third generator
    
'''

import csv, os, glob, sys, getopt

def main():

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:ole:n:p:c:a:r:vs:g:f:1:2:3:t", \
            ['help', 'dataset=', 'eval=', 'cross=', 'absolute=', 'runs=', 'scoretype=', 'verbose', 'generator=', 'file=', 'test', 'number='])
    except getopt.error, msg:
        print msg
        sys.exit(2)
   
    #===========================================================================
    # Process options
    #===========================================================================
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        if o in ("-p", "--problem"):
            p = int(a)
        if o in ("-r", "--runs"): runs = int(a)        
        if o in ("-c", "--cross"):
            cols = eval('[' + a + ']')
            variates = crosscols(cols[1], cols[2])
        if o in ("-a", "--absolute"):
            cols = eval('[' + a + ']')
            variates = range(cols[1], cols[2] + 1)
        if o in ("-p", "--plot"): plot = True; runs = 1
        if o in ("-t", "--test"): test = True  
        if o in ("-d", "--dataset"): dataset = a     
        if o in ("-n", "--number"): nparticles = int(a)   
        if o in ("-f", "--file"): filestub = a
        if o in ("-v", "--verbose"): verbose = True
        if o in ("-s", "--scoretype"):
            scoretype = a
            if not scoretype in ('hb', 'bic'):
                print "Score type " + a + " unknown. Choose \'hb\' or \'bic\'."
                sys.exit(2)
        if o in ("-1", "-2", "-3"): k = eval('[' + a + ']')            
        if o == "-1": param[:len(k)] = k 
        if o == "-2": param[:len(k)] = k 
        if o == "-3": param[:len(k)] = k
        if o in ("-g", "--generator"): gens = a
        if o in ("-e", "--eval"): 
            filename = a[a.rfind("/") + 1:]
            if a == 'all': filename = ''
            ceeval(PATH + "/testruns/" + filename)
            sys.exit(0)
        if o in ("-o", "-l"):
            for o in opts:
                if o in ("-e", "--eval"): break
            if not o in ("-e", "--eval"): sys.exit(0)

    targetDistr = binary_post(cols[0], variates=variates, dataset=dataset, scoretype=scoretype)
    
    cetest(targetDistr=targetDistr, gens=gens, param=param, runs=runs, nparticles=nparticles, test=test, filestub=filestub, plot=plot, verbose=verbose)

if __name__ == "__main__":
    main()