'''
USAGE: smc [OPTIONS]

    -h, --help
                                display help message
                                
    -t, --test
                                test run, do not save results in file
                                
    -e, --eval       =file.txt
                                evaluate result file, create file.eval.txt and file.pdf 
        
    -b, --bridge     =int
                                numer of bridge steps

    -n, --nparticles =int
                                numer of particles

    -k, --kappa     =float
                                prior smoothing parameter
                                    
    -f, --file
                                output file
        
    -r, --runs       =int
                                number of tests to run
                                
    -p, --plot
                                plot steps for one run
        
    -c, --columns    =int,int,int
                                explained column, first and last columns to cross
        
    -s, --scoretype  ={bic,hb}
                                scoretype to use: bic=Bayesian information criterion, hb=hierachical Bayes
        
    -w, --weighted
                                enable weigted averages and regressions
        
    -g, --generator  ={ilm}
                                generators to use: i=independent, m=multinormal, l=logistic regression
    
    -v, --verbose
                                turns direct output off
    
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
from copy import *
import csv, os, glob, sys, getopt, platform
import cProfile
from mcmc import *

## True, if scipy.weave is available.
WEAVE = False
if platform.system() == 'Linux':
    import scipy.weave as weave
    WEAVE = True

## Absolute path to the data directory.
PATH = '/home/cschafer/Documents/Python/workspace/data'

class smc(object):
    '''
    Sequential Monte Carlo class for model choice problems.    
    '''
    
    def dosmc(self, targetDistr, nparticles, kappa, verbose=True, nbridge=None, plot=False):
        '''
        Construct a particle approximation to the target via sequential Monte Carlo methods.
        
        @param targetDistr The target distribution.
        @param proposalDistr The proposal distributions.
        @param nbridge Number of bridge steps between the prior and the target.
        @param nparticles Number of particles.
        @param kappa Prior smoothing parameter: 0 <= kappa <= 1.
        @param plot True, if the steps are to be plotted.
        @param verbose True, if reports to standard are desired.
        '''
               
        start = time.time()
        self.plot = plot
        self.verbose = verbose
        self.kappa = kappa
        self.targetDistr = targetDistr
        self.changeProposalDistr = False
        if nbridge == None: nbridge = 20 + targetDistr.p
        self.rho = arange(0, 1 + 1 / float(nbridge + 1), 1 / float(nbridge + 1))

        self.n = dict([\
                       ['particles', nparticles], \
                       ['bridge', nbridge], \
                       ['resampling', 0], \
                       ['moves', 0], \
                       ['target evals', nparticles]])
        
        if kappa == 0: mcmc_steps = 0.1
        else: mcmc_steps = targetDistr.p * 0.5
        self.estimatePriorDistr(mcmc_steps)

        self.ps = dict([\
                        ['particles', resize(empty(targetDistr.p, dtype=bool), (nparticles, targetDistr.p))], \
                        ['logweights', zeros(nparticles, dtype=float)], \
                        ['logtarget', empty(nparticles, dtype=float)], \
                        ['logproposal', empty(nparticles, dtype=float)], \
                        ['logprior', empty(nparticles, dtype=float)], \
                        ['bridge t-1', zeros(nparticles, dtype=float)], \
                        ['bridge t', zeros(nparticles, dtype=float)], \
                        ['strings', []]])
                      
        for i in range(self.n['particles']):
            self.ps['particles'][i], self.ps['logproposal'][i] = self.priorDistr.rvsplus()
            self.ps['logtarget'] [i] = self.targetDistr.lpmf(self.ps['particles'][i])
            self.ps['strings'].append(bin2str(self.ps['particles'][i]))
            if self.kappa > 0: self.ps['logprior'][i] = self.ps['logproposal'][i]
            self.ps['bridge t-1'][i] = self.logprior(i)
        
        
        #
        # run sequential Monte Carlo scheme
        #
        if not self.verbose: stdout.write("\n" + 101 * " " + "]" + "\r" + "["); progress = 0
        for self.t in range(1, self.n['bridge'] + 1):

            if self.verbose:
                print " %i/%i " % (self.t, self.n['bridge'])
            else:
                progress_next = 100 * self.t / self.n['bridge']
                if 100 * self.n['bridge'] % self.t == 0:
                    stdout.write((progress_next - progress) * "-")
                    stdout.flush()
                    progress = progress_next
            
            # update weights and check effective sample size
            ess = self.updateLogWeights()
            if self.verbose: print "ess: %.3f" % (ess / float(self.n['particles']))

            if ess / float(self.n['particles']) < 0.5 + 0.45 * self.t / self.n['bridge']:
                self.n['resampling'] += 1
                
                self.estimateProposalDistr() 
                self.resampleParticleSystem()
                self.moveParticleSystem()
                
        # return results
        print "\nDone in %.3f seconds.\n" % (time.time() - start)
        pdata = data(dataset=self.ps['particles'], weightset=self.normalize()); pdata.isArray = True
        mprobs = pdata.mean(weighted=True)
        mprobs = '[' + ''.join(map(lambda x: ('%.3f, ' % x), mprobs[:-1])) + ('%.3f]' % mprobs[-1])
        return self.proposalDistr.name, mprobs, self.n['moves'], self.n['target evals'] / 1000., time.time() - start

    def estimatePriorDistr(self, steps):
        '''
        Run a pilot Markov chain to estimate the prior distribution and the log level.
        
        @param steps Number of MCMC steps for the pilot run.
        '''
        if self.verbose: print "run pilot Markov chain -",
        pilot = mcmc()
        if steps == 0.1: singlestep = 100
        else: singlestep = 10000
        result = pilot.domcmc(self.targetDistr, ths_iter=steps, kernel='mh', singlestep=singlestep, verbose=False)
        self.loglevel = result[4]
        if self.kappa == 0:
            mean = "uniform"
        else:
            mprobs = array(eval(result[1]))
            mean = self.kappa * mprobs + (1 - self.kappa) * 0.5 * ones(self.targetDistr.p)
        self.proposalDistr = binary_ind(mean=mean, p=self.targetDistr.p)
        self.priorDistr = binary_ind(mean=mean, p=self.targetDistr.p)
        if self.verbose: print "done."

    def estimateProposalDistr(self):
        '''
        Estimate the parameters of the proposal distribution from the particle system.
        '''
        if self.verbose: print "estimate..."
        if self.proposalDistr.name == 'binary_ind' and self.changeProposalDistr: self.proposalDistr = binary_log()

        # aggregate particle weights for faster estimation of log regession
        if not self.changeProposalDistr == None:
            weights = self.normalize()
            fdata = []; fweights = []
            sorted = argsort(self.ps['strings'])
            particle = self.ps['particles'][sorted[0]]
            weight = weights[sorted[0]]; count = 1
            for index in sorted[1:]:
                if (particle == self.ps['particles'][index]).all():
                    count += 1
                else:
                    fdata.append(particle)
                    fweights.append(weight * count)
                    particle = self.ps['particles'][index]
                    weight = weights[index]
                    count = 1
            
        self.pdata = data(dataset=array(fdata), weightset=array(fweights)); self.pdata.isArray = True
        self.proposalDistr.__init__(self.pdata, fraction_mean=1, fraction_corr=1, \
                                    smooth_mean=0, smooth_corr=0, threshold_randomness=0.03, weighted=True, verbose=self.verbose)
              
        # Note: The log proposal values are updated after the residual resampling step using the fact that they are grouped
          
        if self.verbose: print "strongly random:", self.proposalDistr.p
        if self.plot: self.plotSteps(self.n['resampling'])


    def moveParticleSystem(self):
        '''
        Move particles according to an MH kernel to fight depletion of the particle system.
        '''
        if self.verbose: print "move..."; step = int(self.n['particles'] / 10)
        previous_particleDiversity = 0
        for iter in range(10):
            acceptanceRatio = 0
            self.n['moves'] += 1
            if self.verbose:
                stdout.write("%i " % iter)
                stdout.write("[")
            
            # generate new particles from invariant kernel
            for index in range(self.n['particles']):
                if self.verbose and index % step == 0: stdout.write("-")
                acceptanceRatio += self.kernel_indmh(index)
                
            if self.verbose: print "]"          
            
            particleDiversity = self.getParticleDiversity()

            # check if binary_ind performs poorly and change to binary_log
            if self.proposalDistr.name == 'binary_ind' and acceptanceRatio < self.n['particles'] * 0.25:
                if self.verbose: print "switch to binary_log after next resampling..."
                self.changeProposalDistr = True
            
            if self.verbose:print "acc: %.3f, pdiv: %.3f" % (acceptanceRatio / float(self.n['particles']), particleDiversity)
            if particleDiversity - previous_particleDiversity < 0.05 or particleDiversity > 0.92: break
            else: previous_particleDiversity = particleDiversity

        self.ps['logweights'] = zeros(self.n['particles'])
        self.ps['bridge t-1'] = self.rho[self.t] * self.ps['logtarget'] + (1 - self.rho[self.t]) * self.logprior()
        return particleDiversity


    def kernel_indmh(self, index):
        '''
        Metropolis Hasting kernel with independent proposal distribution estimated from the particle system.
        
        @param index Index of the particle to be processed by the kernel.
        '''
        
        # generate proposal
        proposal, new_proposalScore = self.proposalDistr.rvsplus()
        new_targetScore = self.targetDistr.lpmf(proposal)
        self.n['target evals'] += 1

        if self.kappa > 0: new_priorScore = self.logprior(self.priorDistr.lpmf(proposal))
        else: new_priorScore = self.logprior()
        
        new_bridgeScore = self.rho[self.t] * new_targetScore + (1 - self.rho[self.t]) * new_priorScore
        bridgeScore = self.rho[self.t] * self.ps['logtarget'][index] + (1 - self.rho[self.t]) * self.logprior(index)
        
        # compute acceptance probability and do MH step
        if rand() < exp(self.ps['logproposal'][index] + new_bridgeScore - new_proposalScore - bridgeScore):
            self.ps['particles']  [index] = proposal
            self.ps['strings']    [index] = bin2str(proposal)
            self.ps['logtarget']  [index] = new_targetScore
            self.ps['logproposal'][index] = new_proposalScore
            if self.kappa > 0: self.ps['logprior'][index] = new_priorScore            
            return 1
        else:
            return 0
              
    def updateLogWeights(self):
        '''
        Update the log weights. Return effective sample size.
        '''
        self.ps['bridge t'] = self.rho[self.t] * self.ps['logtarget'] + (1 - self.rho[self.t]) * self.logprior()
        self.ps['logweights'] += self.ps['bridge t'] - self.ps['bridge t-1']
        self.ps['bridge t-1'] = deepcopy(self.ps['bridge t'])
        return self.ess()
    
    def normalize(self):
        '''
        Return normalized importance weights.
        '''
        logweights = self.ps['logweights'] - self.ps['logweights'].max();
        weights = exp(logweights); weights /= sum(weights)
        return weights
    
    def ess(self):
        '''
        Return effective sample size 1/(sum_{w \in weights} w^2) .
        '''
        weights = self.normalize()
        return 1 / pow(weights, 2).sum()
    
    def getParticleDiversity(self):
        '''
        Return the particle diversity.
        '''
        d = {}
        map(setitem, (d,)*len(self.ps['strings']), self.ps['strings'], [])
        return len(d.keys()) / float(self.n['particles'])

    def resampleParticleSystem(self):
        '''
        Resample the particle system.
        '''
        if self.verbose: print "resample..."
        if WEAVE: indices = self.resample_weave()
        else: indices = self.resample_python()

        self.ps['particles'] = self.ps['particles'][indices]
        self.ps['strings'] = [ self.ps['strings'][i] for i in indices ]
        self.ps['logtarget'] = self.ps['logtarget'][indices]
        self.ps['logproposal'] = self.ps['logproposal'][indices]
        
        if self.verbose: print "pdiv: ", self.getParticleDiversity()
        
        # update log proposal/prior values - use that particles are grouped after resampling
        self.ps['logproposal'][0] = self.proposalDistr.lpmf(self.ps['particles'][0])
        if self.kappa > 0:
            self.ps['logprior'][0] = self.priorDistr.lpmf(self.ps['particles'][0])
        for i in range(1, self.n['particles']):
            if (self.ps['logproposal'][i] == self.ps['logproposal'][i - 1]).all():
                self.ps['logproposal'][i] = self.ps['logproposal'][i - 1]
                if self.kappa > 0:
                    self.ps['logprior'][i] = self.priorDistr.lpmf(self.ps['logprior'][i - 1])
            else:
                self.ps['logproposal'][i] = self.proposalDistr.lpmf(self.ps['particles'][i])
                if self.kappa > 0:
                    self.ps['logprior'][i] = self.priorDistr.lpmf(self.ps['particles'][i])

    
    def logprior(self, x=None):
        '''
        Return the prior log probability.
        
        @param x
        If x is an index, the function returns self.ps['logprior'][x] plus the log level.
        If x is a float, it returns x plus the log level.
        If x is not specified, it returns the vector self.ps['logprior'] plus the loglevel.
        '''
        if self.kappa == 0:
            return self.loglevel - self.targetDistr.p * log(2)
        if type(x).__name__ == "int":
            return self.ps['logprior'][x] + self.loglevel
        if type(x).__name__ == "float64":
            return x + self.loglevel
        if x == None:
            return self.ps['logprior'] + self.loglevel
        
    
    def resample_python(self):
        '''
        Compute the particle indices by residual resampling - adopted from Pierre's code.
        '''
        w = self.normalize()
        u = random.uniform(size=1, low=0, high=1)
        cnw = self.n['particles'] * cumsum(w)
        j = 0
        indices = empty(self.n['particles'], dtype="int")
        for k in xrange(self.n['particles']):
            while cnw[j] < u:
                j = j + 1
            indices[k] = j
            u = u + 1.
        return indices
    
    def resample_weave(self):
        '''
        Compute the particle indices by residual resampling using scypy.weave.
        '''
        code = \
        """
        int j = 0;
        double cumsum = weights(0);
        
        for(int k = 0; k < n; k++)
        {
            while(cumsum < u)
            {
            j++;
            cumsum += weights(j);
            }
            indices(k) = j;
            u = u + 1.;
        }
        """
        u = float(random.uniform(size=1, low=0, high=1)[0])
        n = self.n['particles']
        weights = n * self.normalize()
               
        indices = zeros(self.n['particles'], dtype="int")
        weave.inline(code, ['u', 'n', 'weights', 'indices'], \
                     type_converters=weave.converters.blitz, compiler='gcc')
        return indices

        
    def plotSteps(self, step):
        '''
        Plot the marginal probabilities and correlation matrix of the particle system.
        '''
        particle_pi = self.pdata.mean(weighted=True)
        particle_R = self.pdata.cor(weighted=True)
        samp = sampler(self.proposalDistr)
        samp.sample(self.n['particles'], verbose=False)     
        proposal_pi = samp.data.mean()
        proposal_R = samp.data.cor()
        plotSteps(left=particle_pi, right=proposal_pi, step=step, diff=False)
        plotSteps(left=particle_R, right=proposal_R, step=step, diff=False)
    
    def printParticleStructure(self):
        '''
        Print out a summary of how many particles are n-fold in the particle system.
        '''
        s = set(self.ps['strings'])
        l = [ self.ps['strings'].count(str) for str in s ]
        k = [ l.count(i) * i for i in range(1, 101) ]
        print k, sum(k)

def smctest(targetDistr, filestub, runs, nbridge, nparticles, kappa, test, plot, verbose):
    '''
    Run repeated tests of bridge smc.
    
    @param targetDistr The target distribution.
    @param filestub Name of the test-file without numbering.
    @param runs Number of testruns to perform.
    @param nbridge Number of bridge steps between the prior and the target.
    @param nparticles Number of particles.
    @param kappa Prior smoothing parameter: 0 <= kappa <= 1.
    @param test True, if no testfile is desired.
    @param plot True, if the steps are to be plotted.
    @param verbose True, if reports to standard are desired.
    '''

    # create summary and header for test report
    if nbridge == None: nbridge = 20 + targetDistr.p
    summary = "kappa=%.2f, bridge=%i, particles=%i, psize=%i" % (kappa, nbridge, nparticles, targetDistr.p)
    summary += ", dataset=" + targetDistr.dataset + ", score=" + targetDistr.scoretype
    
    if not test:
        if filestub == None:
            filestub = "%xsmc" % (long(datetime.now().strftime("%m%d%H%M")))
        elif filestub.rfind("smc") == -1:
            filestub = filestub + 'smc'
        
        filename = "../../data/testruns/" + filestub + "_1.txt"; i = 1
        while os.path.exists(filename):
            filename = "../../data/testruns/" + filestub + ("_%i.txt" % i)
            i += 1
        file = open(filename, 'w')
        file.write(summary + "\ntype;mprobs;moves;targetevals;time\n")
        file.close()
    
    # do test runs
    testsmc = smc()
    for iter in range(runs):
        print ("%i/%i " % (iter + 1, runs)) + summary
        result = testsmc.dosmc(targetDistr=targetDistr, nparticles=nparticles, nbridge=nbridge, verbose=verbose, kappa=kappa, plot=plot)
        
        # write results to file
        if not test: file = open(filename, 'a')
        for item in result:
            print str(item) + ";",
            if isinstance(item, list): item = repr(item)
            if not test: file.write(str(item) + ";")
        print "\n"
        if not test: file.write("\n"); file.close()
            
def smceval(filestubs='*', boxdata=0.8, fragmaster=False):
    '''
    Make pdf-boxplots from testfiles.
    
    @param filename Name of the testfile to be processed.
    @param boxdata Percentage of data to be included in the core of the boxplot.
    '''
    
    evaldict = dict([\
        ['header', ['', '']], \
        ['hist', [[], []]], \
        ['targetevals', array([0., 0.])], \
        ['time', array([0., 0.])], \
        ['n', array([0, 0])], \
        ['p', array([0, 0])], \
        ['quantiles', [0, 0]]\
        ])
    
    cols = []
    for ifilestub, filestub in enumerate(filestubs):
        files = glob.glob(PATH + '/testruns/' + filestub + '_*.txt')
        if len(files) == 0:
            print "No results files in " + filestub + "_*.txt."
            break

        datreader = csv.reader(open(files[0]), delimiter=';')
        evaldict['header'][ifilestub] = datreader.next()[0]
        row = datreader.next()
        cols.append(dict([[x, i] for i, x in enumerate(row)]))       

        # process all files belonging to one filestub
        print "Processing " + filestub + "..."        
        for file in files:
            datreader = csv.reader(open(file), delimiter=';')
            datreader.next(); datreader.next()

            while True:
                try:
                    row = datreader.next()
                    evaldict['hist'][ifilestub].append(array(eval(row[cols[ifilestub]['mprobs']])))
                    evaldict['targetevals'][ifilestub] += float(row[cols[ifilestub]['targetevals']])
                    evaldict['time'][ifilestub] += float(row[cols[ifilestub]['time']])
                except:
                    break

        evaldict['n'][ifilestub] = len(evaldict['hist'][ifilestub])
        evaldict['p'][ifilestub] = len(evaldict['hist'][ifilestub][0])
        
        # create box plot
        histArray = empty((evaldict['n'][ifilestub], evaldict['p'][ifilestub]))
        evaldict['quantiles'][ifilestub] = zeros((5, evaldict['p'][ifilestub]))
        for i, mprobs in enumerate(evaldict['hist'][ifilestub]):
            histArray[i] = mprobs
        histArray.sort(axis=0)
        for i in range(evaldict['p'][ifilestub]):
            evaldict['quantiles'][ifilestub][0][i] = histArray[:, i][0]
            for j, q in [(1, 1. - boxdata), (2, 0.5), (3, boxdata), (4, 1.)]:
                evaldict['quantiles'][ifilestub][j][i] = histArray[:, i][int(q * evaldict['n'][ifilestub]) - 1] - \
                evaldict['quantiles'][ifilestub][:j + 1, i].sum()
       
        n = float(evaldict['n'][ifilestub])
        evaldict['targetevals'][ifilestub] = evaldict['targetevals'][ifilestub] / n
        evaldict['time'][ifilestub] = evaldict['time'][ifilestub] / n
        
    # plot with rpy
    r.pdf(paper="a4r", file=PATH + "/testruns/" + ''.join(filestubs) + ".pdf", width=12, height=12)
    colors = ['grey85', 'black', 'white', 'white', 'black']
    if fragmaster:
        r.par(mfrow=[len(filestubs),1], oma=[0, 0, 0 , 0], mar=[2, 0, 2, 0])
    else:
        r.par(mfcol=[len(filestubs),1], oma=[0, 0, 2 , 0], mar=[2, 0, 2, 0])        
    for ifilestub in range(len(filestubs)):
        if fragmaster:
            title = "title " + str(ifilestub)
        else:
            title = filestubs[ifilestub] + ": ths. evals=%.1f, time=%.3f" % \
                (evaldict['targetevals'][ifilestub], evaldict['time'][ifilestub]) + \
                ", " + str("boxdata=%i%%, test runs=%i," % (boxdata * 100, n)) + evaldict['header'][ifilestub]
        r.barplot(evaldict['quantiles'][ifilestub], ylim=[0, 1], axes=False, col=colors)
        r.title(main=title, family="serif", cex_main=1, font_main=1)
    r.dev_off()

def main():
    '''
    Parse command line options to smctest. Type smc --help for help.
    '''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:b:olk:e:f:n:c:a:r:vs:g:tp", \
            ['boxdata=', 'help', 'dataset=', 'plot', 'bridge=', 'eval=', 'cross=', 'absolute=', 'runs=', \
             'scoretype=', 'verbose', 'generator=', 'test', 'nparticles=', 'frag'])
    except getopt.error, msg:
        print msg
        sys.exit(2)
    
    # Set default parameters.
    test = False
    plot = False
    cols = [8, 9, 21]
    variates = crosscols(cols[1], cols[2])
    runs = 200
    nparticles = 20000
    nbridge = None
    verbose = False
    kappa = 0
    scoretype = 'hb'
    proposalDistrs = []
    genstr = 'l'
    dataset = 'boston'
    boxdata = 0.8
    filestub = None
    fragmaster = False

    # Process options.
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        if o in ("-p", "--problem"):
            p = int(a)
        if o == "--boxdata": boxdata = float(a)
        if o in ("-r", "--runs"): runs = int(a)
        if o in ("-c", "--cross"):
            cols = eval('[' + a + ']')
            variates = crosscols(cols[1], cols[2])
        if o in ("-a", "--absolute"):
            cols = eval('[' + a + ']')
            variates = range(cols[1], cols[2] + 1)
        if o in ("-t", "--test"): test = True
        if o in ("-p", "--plot"): plot = True; runs = 1
        if o in ("-k", "--kappa"): kappa = float(a)
        if o in ("-d", "--dataset"): dataset = a     
        if o in ("-n", "--nparticles"): nparticles = int(a)   
        if o in ("-b", "--bridge"): nbridge = int(a)
        if o in ("-f", "--file"): filestub = a
        if o == '--frag': fragmaster = True
        if o in ("-v", "--verbose"): verbose = True
        if o in ("-s", "--scoretype"):
            scoretype = a
            if not scoretype in ('hb', 'bic'):
                print "Score type " + a + " unknown. Choose \'hb\' or \'bic\'."
                sys.exit(2)
        if o in ("-o", "-l"):
            for o in opts:
                if o in ("-e", "--eval"): break
            if not o in ("-e", "--eval"): sys.exit(0)
            
    for o, a in opts:
        if o in ("-e", "--eval"):
            filestubs = eval("['" + a.replace(",", "','") + "']")
            smceval(filestubs=filestubs, boxdata=boxdata, fragmaster=fragmaster)
            sys.exit(0)
    
    targetDistr = binary_post(cols[0], variates=variates, dataset=dataset, scoretype=scoretype)
    smctest(targetDistr=targetDistr, filestub=filestub, runs=runs, nparticles=nparticles, \
            plot=plot, nbridge=nbridge, kappa=kappa, verbose=verbose, test=test)

if __name__ == "__main__":
    #cProfile.run('main()', filename='prof')
    main()
