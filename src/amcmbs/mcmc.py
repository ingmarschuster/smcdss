'''
USAGE: mcmc [OPTIONS]

    -h, --help
                                display help message
                                
    -t, --test
                                test run, do not save results in file
                                
    -e, --eval       =file.txt
                                evaluate result file, create file.eval.txt and file.pdf 
        
    -k, --kernel     ={gibbs,mh},int
                                kernel driving the Markov chain

    -d, --dataset
                                dataset to be used
                                
    -m, --maxtime    =int
                                maximum time in minutes

    -i, --ths_iter =int
                                number of 1.000s of ths_iter to run
        
    -r, --runs       =int
                                number of tests to run
                                
    -p, --plot
                                plot steps for one run
                                        
    -v, --verbose
                                turns direct output off

    -f, --file
                                output file
        
    -c, --columns    =int,int,int
                                explained column, first and last columns to cross
        
    -s, --scoretype  ={bic,hb}
                                scoretype to use: bic=Bayesian information criterion, hb=hierachical Bayes
    
'''


from gen import *
from numpy import *
from sampling import *
try:
    from plotting import *
except:
    print 'Can\'t import rpy for plotting.'

from datetime import datetime
from operator import setitem
from copy import copy, deepcopy
from scipy.stats.distributions import inf
import csv, os, glob, sys, getopt

PATH = '/home/cschafer/Documents/Python/workspace/data'

class mcmc(object):
    def domcmc(self, targetDistr=None, maxtime=inf, ths_iter=inf, verbose=False, \
               kernel='mh', singlestep=10000, plot=False):
        '''
        Runs a MCMC sampling scheme to approximate the marginal distribution.
                
        '''
        
        self.verbose = verbose
        
        if self.verbose:print "start markov chain with " + kernel + " kernel." 
                        
        start = time.time()
        self.plot = plot
        self.targetDistr = targetDistr
        state = rand(targetDistr.p) < 0.5 * ones(targetDistr.p)
        self.score = self.targetDistr.lpmf(state)
        
        if kernel == 'gibbs':
            self.kernel = self.kernel_gibbs
        else:
            self.kernel = self.kernel_indmh
            self.scaleneighbors = 0 #max(targetDistr.p / 20., 1)
        
        # burn in period
        self.maxval = -inf
        self.state_change = 0
        for i in range(singlestep):
            state = self.kernel(state)
    
        # plot final probabilities
        if plot: self.plotSteps(self.nresampling + 1)

        pdata = zeros(self.targetDistr.p)
        step_iter = 0; mc_iter = 0; self.state_change = 0
        
        if self.verbose:stdout.write("\n" + 101 * " " + "]" + "\r" + "["); progress = 0           
        while (time.time() - start) / 60. < maxtime and mc_iter < ths_iter * 1000:
            sdata = data();
            for i in range(singlestep):
                mc_iter += 1
                state = self.kernel(state)
                sdata.append(state)
            pdata = step_iter * pdata + sdata.mean()
            step_iter += 1

            if self.verbose:
                if maxtime == inf:
                    if mc_iter >= ths_iter * 1000: progress_next = 100
                    progress_next = int(0.1 * mc_iter / float(ths_iter))
                    stdout.write((progress_next - progress) * "-")
                    stdout.flush()
                    progress = progress_next
                else:
                    progress_next = int(10 * (time.time() - start) / float(6 * maxtime))
                    if (time.time() - start) / 60. >= maxtime: progress_next = 100
                    stdout.write((progress_next - progress) * "-")
                    stdout.flush()
                    progress = progress_next
                
            pdata /= float(step_iter)

        # return results
        if verbose: print
        mprob = '[' + ''.join(map(lambda x: ('%.3f, ' % x), pdata[:-1])) + ('%.3f]' % pdata[-1])
        return kernel, mprob, mc_iter / 1000., 100 * self.state_change / float(mc_iter), time.time() - start, self.maxval
        
    # Gibbs kernel driving the Markov chain
    def kernel_gibbs(self, state):
        proposal = deepcopy(state)
        index = random.randint(0, self.targetDistr.p)
        proposal[index] = proposal[index] ^ True
        proposalscore = self.targetDistr.lpmf(proposal)
        if rand() < 1 / (1 + exp(self.score - proposalscore)):
            self.score = proposalscore
            self.state_change += 1
            return proposal
        else:
            return state
        
    # MH kernel driving the Markov chain
    def kernel_indmh(self, state):
        if self.scaleneighbors > 0:
            components = set()
            v = random.geometric(1 / float(self.scaleneighbors))
            while len(components) < min(v, self.targetDistr.p / 2):
                components.add(random.randint(0, self.targetDistr.p))
        else:
            components = [random.randint(0, self.targetDistr.p)]
                    
        proposal = deepcopy(state)
        for index in components:
            proposal[index] = proposal[index] ^ True
        
        proposalscore = self.targetDistr.lpmf(proposal)
        if rand() < exp(proposalscore - self.score):
            if proposalscore > self.maxval: self.maxval = proposalscore
            self.score = proposalscore
            self.state_change += 1
            return proposal
        else:
            return state
        
    def printMarginalProbs(self):
        self.pdata = data(dataset=self.particles, weightset=self.normalize()); self.pdata.isArray = True
        print "\nmarginal prob\'s:\n", array([round(x, 3) for x in self.pdata.mean(weighted=True)])
        
    # Plot the marginal probabilities and correlation matrix of the particle system
    def plotSteps(self, step):
        particle_pi = self.pdata.mean(weighted=True)
        particle_R = self.pdata.cor(weighted=True)
        samp = sampler(self.proposalDistr)
        samp.sample(self.nparticles, verbose=False)     
        proposal_pi = samp.data.mean()
        proposal_R = samp.data.cor()
        plotSteps(left=particle_pi, right=proposal_pi, step=step, diff=False)
        plotSteps(left=particle_R, right=proposal_R, step=step, diff=False)
       
def mcmctest(targetDistr, filestub, runs, kernel, test, plot, verbose, maxtime=inf, ths_iter=inf):
    '''
    Run repeated tests of bridge smc.
    '''
    
    # create summary and header for test report
    if ths_iter == inf:
        summary = "maxtime=%.1f, psize=%i" % (maxtime, targetDistr.p)
    else:
        summary = "iter=%i, psize=%i" % (ths_iter, targetDistr.p)
    
    summary += ", dataset=" + targetDistr.dataset + ", score=" + targetDistr.scoretype
      
    if not test:
        if filestub == None:
            filestub = "%xmcmc" % (long(datetime.now().strftime("%m%d%H%M")))
        elif filestub.rfind("mcmc") == -1:
            filestub = filestub + 'mcmc'
            
        filename = "../../data/testruns/" + filestub + "_1.txt"; i = 1
        while os.path.exists(filename):
            filename = "../../data/testruns/" + filestub + ("_%i.txt" % i)
            i += 1
        file = open(filename, 'w')
        file.write(summary + "\ntype;mprobs;targetevals;accratio;time;maxeval\n")
        file.close()
    
    # do test runs
    testmcmc = mcmc()
    for iter in range(runs):
        print ("%i/%i " % (iter+1, runs)) + summary
        # run Markov chain
        result = testmcmc.domcmc(targetDistr=targetDistr, maxtime=maxtime, ths_iter=ths_iter, \
                                 kernel=kernel, verbose=verbose, plot=plot)
        # write results to file
        if not test: file = open(filename, 'a')
        for item in result:
            if verbose: print str(item) + ";",
            if isinstance(item, list): item = repr(item) 
            if not test: file.write(str(item) + ";")
        if verbose: print "\n"
        if not test: file.write("\n"); file.close()
            
def mcmceval(filename=PATH + '/testruns', boxdata=0.8):
    '''
    Plot test run quantiles as .pdf-files.  
    '''
    try:
        open(filename)
        files = [filename]
    except:
        files = glob.glob(os.path.join(filename, '*mcmc_1.txt'))
        if len(files) == 0:
            print "No results files in " + filename + "."
            return

    for leadfile in files:
        nosuffix = leadfile[leadfile.rfind("/") + 1:-6]
        print "Processing " + nosuffix + "..." 
        datreader = csv.reader(open(leadfile), delimiter=';')
        histList = [[], []]; evals = array([0., 0.]);accrate = array([0., 0.]); times = array([0., 0.])
                
        # find model types
        datreader.next(); datreader.next()
        types = []
        #for i in range(1, -1, -1):
        types.append([datreader.next()[0], 0])
        types = dict(types)

        # read simulation results        
        for filename in glob.glob(os.path.join(leadfile[:leadfile.rfind("/")], '*' + nosuffix + '*.txt')):
            datreader = csv.reader(open(filename), delimiter=';')
            header = datreader.next()[0]; row = datreader.next()

            while True:
                try:
                    rows = []
                    for i in range(len(types)):
                        row = datreader.next()
                        while len(row) == 0:
                            row = datreader.next()                            
                        rows.append(row)
                except:
                    break
                for row in rows:
                    histList[types[row[0]]].append(array(eval(row[1])))
                    evals[types[row[0]]] += float(row[2])
                    accrate[types[row[0]]] += float(row[3])
                    times[types[row[0]]] += float(row[4])
        n = len(histList[0])
        p = len(histList[0][0])

        # create box plot      
        qArray = []
        histArray = []
        for type in range(len(types)):
            histArray.append(empty((n, p)))
            qArray.append(zeros((5, p)))
            for i, mprob in enumerate(histList[type]):
                histArray[type][i] = mprob
            histArray[type].sort(axis=0)
            for i in range(p):
                qArray[type][0][i] = histArray[type][:, i][0]
                for j, q in [(1, 1. - boxdata), (2, 0.5), (3, boxdata), (4, 1.)]:
                    qArray[type][j][i] = histArray[type][:, i][int(q * n) - 1] - qArray[type][:j + 1, i].sum()
        
        times = times / float(n); evals = evals / float(n)

        # plot with rpy
        r.pdf(paper="a4r", file=PATH + "/testruns/" + nosuffix + ".pdf", width=12, height=12)
        colors = [['grey85', 'skyblue', 'gold', 'gold2', 'skyblue'], ['grey85', 'skyblue', 'green2', 'green3', 'skyblue']]
        r.par(mfrow=[1, len(types)], oma=[0, 0, 4, 0], mar=[2, 0, 2, 0])
        for i in range(len(types)):
            r.barplot(qArray[i], ylim=[0, 1], axes=False, col=colors[i])
            r.title(main=types.keys()[i] + ": ths. evals=%.1f, acc rate=%.1f %%, time=%.3f" % (evals[i], accrate[i], times[i]),family="serif")
        r.mtext("file=" + nosuffix + str(", boxdata=%i%%, test runs=%i," % (boxdata * 100, n)) + header, outer=True, line=1, cex=1, family="serif")
        r.dev_off()

def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:ole:c:a:r:vs:f:k:tp:m:i:", \
            ['boxdata=', 'help', 'dataset=', 'maxtime=', 'ths_iter=', 'plot', 'eval=', 'cross=', 'absolute=', 'runs=', 'scoretype=', 'verbose', 'kernel=', 'test'])
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
    kernel = 'mh'
    verbose = True
    scoretype = 'hb'
    dataset = 'boston'
    boxdata = 0.8
    maxtime = 30
    ths_iter = inf
    filestub = None
    neighbors = None

    #===========================================================================
    # Process options
    #===========================================================================
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        if o in ("-p", "--problem"):
            p = int(a)
        if o == "--boxdata": boxdata = float(a)
        if o in ("-r", "--runs"): runs = int(a)
        if o in ("-f", "--file"): filestub = a
        if o in ("-c", "--cross"):
            cols = eval('[' + a + ']')
            variates = crosscols(cols[1], cols[2])
        if o in ("-a", "--absolute"):
            cols = eval('[' + a + ']')
            variates = range(cols[1], cols[2] + 1)
        if o in ("-t", "--test"): test = True
        if o in ("-p", "--plot"): plot = True; runs = 1
        if o in ("-d", "--dataset"): dataset = a     
        if o in ("-k", "--kernel"): kernel = str(a)
        if o in ("-v", "--verbose"): verbose = False
        if o in ("-m", "--maxtime"):
            maxtime = float(a); ths_iter = inf
        if o in ("-i", "--ths_iter"):
            maxtime = inf; ths_iter = float(a)
        if o in ("-s", "--scoretype"):
            scoretype = a
            if not scoretype in ('hb', 'bic'):
                print "Score type " + a + " unknown. Choose \'hb\' or \'bic\'."
                sys.exit(2)
        if o in ("-e", "--eval"): 
            filename = a[a.rfind("/") + 1:]
            if a == 'all': filename = ''
            mcmceval(PATH + "/testruns/" + filename, boxdata=boxdata)
            sys.exit(0)
        if o in ("-o", "-l"):
            for o in opts:
                if o in ("-e", "--eval"): break
            if not o in ("-e", "--eval"): sys.exit(0)

    targetDistr = binary_post(cols[0], variates=variates, dataset=dataset, scoretype=scoretype)
    mcmctest(targetDistr=targetDistr, filestub=filestub, runs=runs, maxtime=maxtime, ths_iter=ths_iter, kernel=kernel, \
              test=test, plot=plot, verbose=verbose)
            
if __name__ == "__main__": main()
