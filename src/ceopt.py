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

from gen import *
from sampling import *
try:
    from plotting import *
except:
    print 'Can\'t import rpy for plotting.'

from datetime import datetime
from copy import copy
import csv, os, glob, sys, getopt

TEST_PATH = '/home/cschafer/Documents/Python/workspace/data/testruns'
PDF_PATH = '/home/cschafer/Tex/amc_on_mbs/img/fragmaster'
    
def ceopt(targetDistr, gens, param, nparticles, plot, verbose):
    '''
    Run cross entropy optimization.
    '''
    
    state_max = 0
    score_max = -inf
    
    for letter in gens:
        if letter == "i": gen = binary_ind()
        if letter == "m": gen = binary_mn()
        if letter == "l": gen = binary_log()
    gen.__init__(fraction_mean=param[0], fraction_corr=param[1], \
                smooth_mean=param[2], smooth_corr=param[3], \
                threshold_randomness=0.02, min_p=12)
    
    #
    # presampling step
    #
    start = time.clock()
    if verbose: print "\npresampling using binary_ind generator:\nstep 0",
    
    # uniform presampling
    pre = binary_ind(mean="uniform", p=targetDistr.p, verbose=verbose)
    samp = sampler(gen=pre, targetDistr=targetDistr)
    evals = samp.sample(nparticles, verbose=verbose, online=True)
    
    # switch to main generator
    samp.gen = gen
    gen.reset(data=samp.data)
    fraction = gen.fraction_mean

    # remove all but elite samples
    samp.data.clear(fraction=fraction)
    if verbose: print "score: %.5f\n" % samp.data.weightset[0]
    
    # save marginals for plotting
    if not plot == None:
        elite_pi = samp.data.mean()
        elite_R = samp.data.cor()
    
    #
    # run crosss entropy optimization scheme
    #
    if verbose: print "cross entropy optimization using " + gen.name + " generator:"
    for step in range(1, 51):
        if verbose: print "step %i" % step,
        
        # create weighted sample
        evals += samp.sample(nparticles, verbose=verbose, online=gen.dim == gen.p)

        # plot marginal prob's and correlation matrices
        if plot:
            sampled_pi = samp.data.mean(fraction=1)
            sampled_R = samp.data.cor(fraction=1)
            plotSteps(left=elite_R, right=sampled_R, step=step, diff=True)
            plotSteps(left=elite_pi, right=sampled_pi, step=step, diff=True)
        
        # reinit sampler with elite samples
        samp.gen.reset(data=samp.data)
        if verbose:
            x = array(samp.gen.weakly_random); y = samp.gen.weakly_random
            print "state: " , list(x[gen.mean[y] > 0.5]), list(x[gen.mean[y] < 0.5]), samp.gen.strongly_random
        
        # if dimension is reduced to feasible size, run brute force search
        if samp.gen.p < samp.gen.min_p or step > 40:
            mode, score = cebrutesearch(gen=gen, targetDistr=targetDistr, state_max=state_max, score_max=score_max)
            time_elapsed = "%.3f" % (time.clock() - start)
            return [score, mode, time_elapsed, evals]
        
        # remove all but elite samples
        samp.data.clear(fraction=fraction)
        state_max = samp.data.dataset[0]
        score_max = samp.data.weightset[0]
                
        if not plot == None:
            elite_pi = samp.data.mean()
            elite_R = samp.data.cor()
        if verbose:
            if not gen.name == "binary_ind": print "report:", gen.adjusted
            print "score: %.5f\n" % samp.data.weightset[0]
  
def cebrutesearch(gen, targetDistr, state_max, score_max= -inf):
    '''
    Find the highest score and corresponding model by brute force search. 
    '''
    where_1 = list(set(where(gen.mean > 0.5)[0]) & set(gen.weakly_random))
    for d in range(2 ** gen.p):
        if gen.p > 0:
            b = gen._expand_01(rv=dec2bin(d, gen.p), where_1=where_1)
        else:
            b = gen.rvs()
        eval = targetDistr.lpmf(b)
        if eval > score_max:
            state_max = b; score_max = eval
        
    model = where(array(state_max) == True)[0]
    return '[' + ''.join(map(lambda x: str(x) + ', ', model[:-1])) + ('%i]' % model[-1]), score_max 
    
def cetest(targetDistr, gens, param, runs, nparticles, filestub, test, plot, verbose):
    '''
    Run repeated tests of ce optimization.
    '''
    # create summary and header for test report
    summary = "test runs=%i, samples=%i, problem size=%i" % (runs, nparticles, targetDistr.p)
    summary += ", dataset=" + targetDistr.dataset + ", score type=" + targetDistr.scoretype
    print summary
    
    if not test:
        if filestub == None:
            filestub = "%xceopt" % (long(datetime.now().strftime("%m%d%H%M")))
        elif filestub.rfind("ceopt") == -1:
            filestub = filestub + 'ceopt'
        
        filename = "../../data/testruns/" + filestub + "_1.txt"; i = 1
        while os.path.exists(filename):
            filename = "../../data/testruns/" + filestub + ("_%i.txt" % i)
            i += 1
        file = open(filename, 'w')
        file.write(summary + "\nscore;mode;time;evals\n")
        file.close()
    
    # do test runs
    for iter in range(runs):
                
        # init and run cross entropy method
        result = ceopt(targetDistr=targetDistr, gens=gens, param=param, nparticles=nparticles, plot=plot, verbose=verbose)
        
        # write results to file
        if not test: file = open(filename, 'a')
        for item in result:
            if verbose: print str(item) + ";",
            if isinstance(item, list): item = repr(item) 
            if not test: file.write(str(item) + ";")
        if verbose: print "\n"
        if not test: file.write("\n"); file.close()
    
def ceeval(filename, color=False, verbose=False):
    '''
    Convert test runs into .eval-files and plots the respective histograms as .pdf-files.  
    '''
    try:
        open(filename)
        files = [filename]
    except:
        files = glob.glob(os.path.join(filename, '*ceopt*.txt'))
        if len(files) == 0:
            print "No results files in " + filename + "."
            return

    # find model types
    datreader = csv.reader(open(files[0]), delimiter=';')
    datreader.next(); datreader.next()
    gens = []
    for i in range(3):
        gens.append([datreader.next()[0], i])
    types = dict(gens)

    models = []; scores = []; hist = []; steps = array([0., 0., 0.]); times = array([0., 0., 0.])
    for filename in files:
        print "Processing " + filename + "..." 
        
        datreader = csv.reader(open(filename), delimiter=';')
        header = datreader.next()[0]; row = datreader.next()
        
        # read simulation results
        for row in datreader:
            if len(row) == 0: continue
            score = float(row[1]); gen = types[row[0]]
            if len(scores) == 0 or (not score in scores):
                scores.append(score); hist.append(array([0., 0., 0.])); models.append(eval(row[3]))
            hist[scores.index(score)][gen] += 1
            steps[gen] += float(row[2]); times[gen] += float(row[4])
    scores = array(scores)
    index = scores.argsort()[::-1]
    hist = array(hist)[index]
    n = hist[:, 0].sum()
    ylim = hist.max()
    for item in [steps, times]: item /= float(n)
        
    # list models and corresponding scores
    evaltxt = "score and model\n"
    for ix in index:
        evaltxt += "%.3f : " % scores[ix]
        evaltxt += str(models[ix]) + "\n"

    # compute model overlap
    evaltxt += "\nmodel overlap:\n    " + ''.join([("%3i " % i) for i in range(len(scores))]) + "\n"
    cont = zeros((len(scores), len(scores)))
    for ix in range(len(scores)):
        for varx in models[index[ix]]:
            for iy in range(len(scores)):
                for vary in models[index[iy]]:
                    if varx == vary:
                        cont[ix][iy] += 1
        evaltxt += ("%3i " % ix) + ''.join([("%3i " % i) for i in (cont[ix, :])]) + "\n"
    
    # compute relative model overlap
    evaltxt += "\nrelative overlap:\n    " + ''.join([("%3i " % i) for i in range(len(scores))]) + "\n"
    cont /= sqrt(dot(cont.diagonal()[:, newaxis], cont.diagonal()[newaxis, :]))
    for ix in range(len(scores)):
        evaltxt += ("%3i " % ix) + ''.join([("%3i " % i) for i in 100 * cont[ix, :]]) + "\n"
    if verbose:
        print evaltxt
    
    # extract filenames
    filestub = filename[filename.rfind("/") + 1:len(filename) - 6]
    file = open(TEST_PATH + "/" + filestub + ".eval", 'w')
    file.write(evaltxt); file.close()

    # plot with rpy
    r.pdf(paper="a4r", file=PDF_PATH + "/" + filestub + ".pdf", width=12, height=12)
    
    if color:
        colors = ['yellow', 'lightblue', 'lightgreen']
    else:
        colors = ['grey85', 'grey75', 'grey65']
    
    if titles:
        r.par(mfrow=[2, 2], oma=[0, 0, 0, 0], mar=[2, 2, 2, 2], family="serif")
    else:
        r.par(mfrow=[2, 2], oma=[0, 0, 4, 0], mar=[2, 2, 4, 2], family="serif")
    for igen, gen in enumerate(gens):
        gen[0] = gen[0].replace("fraction", "f"); gen[0] = gen[0].replace("smooth", "s")
        if titles:
            title = 'title ' + igen
        else:
            title = gen[0] + ": steps %.1f, time %.3f" % (steps[gen[1]], times[gen[1]])
        r.barplot(hist[:, gen[1]], ylim=[0, ylim], axes=False, \
                  names=range(1, len(scores) + 1), cex_names=3. / sqrt(len(scores) + 2), las=1, col=colors[gen[1]])
        r.title(main=title, cex_main=1, font_main=1)
    if not titles:
        r.mtext(str("test runs=%i/" % n) + header[10:], outer=True, line=1, cex=1)
    r.dev_off()
    
def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:ole:n:p:c:a:r:vs:g:f:1:2:3:t", \
            ['help', 'dataset=', 'eval=', 'cross=', 'absolute=', 'runs=', 'scoretype=', 'verbose', 'generator=', 'file=', 'test', 'number='])
    except getopt.error, msg:
        print msg
        sys.exit(2)
    
    #===========================================================================
    #  Default parameters
    #===========================================================================
    
    test = False
    plot = False
    cols = [8, 9, 21]
    variates = crosscols(cols[1], cols[2])
    runs = 200
    nparticles = 10000
    verbose = False
    scoretype = 'hb'
    gens = 'i'
    dataset = 'boston'
    # fraction mean, fraction corr, smooth mean, smooth corr
    param = [0.02, 0.2, 0.3, 0.2]
    threshold_randomness = 0.02
    filestub = None
    min_p = 12
    
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
