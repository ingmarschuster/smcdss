'''
Created on 18 fevr. 2010

@author: cschafer
'''

from datetime import datetime
from numpy import arange
from rpy import *
from sampling import *
from gen import *

PATH = '/home/cschafer/Documents/Python/workspace/data'

def plotSteps(left, right, step, plotdir="pdf/step", diff=False):
    '''
    Plots the marginal probabilites and correlation matrix of each step during ce optimization.
    '''
    
    GRID = 100.
    if len(left.shape) == 2: type = 'R'
    else: type = 'Pi'
    plotdir = plotdir + type
    
    # write header for bash script
    stamp = "%x" % (long(datetime.now().strftime("%m%d%H%M")))
    if step == 1:
        file = open(PATH + "/steps/.steps" + type + ".sh", "w")
        file.write("#!/bin/sh\nif [ \"$1\" = \"-run\" ]\nthen\n okular " + PATH + "/steps/" + stamp + type + ".pdf --presentation &\n exit\nfi\n " + \
                   "gs -q -sPAPERSIZE=letter -dNOPAUSE -dBATCH -sDEVICE=pdfwrite " + \
                   "-sOutputFile=" + PATH + "/steps/" + stamp + type + ".pdf " + PATH + "/steps/" + plotdir + "1.pdf")
    else:
        file = open(PATH + "/steps/.steps" + type + ".sh", "a"); file.write(" " + PATH + "/steps/" + plotdir + "%i.pdf" % step)
    file.close()

    # compute matrix entries
    if type == 'R':
        p = len(left[0])
        leftPlot = (left, 'elite correlation matrix', "yellow", [-1.0, 1.0])
        if diff: rightPlot = (left - right, 'deviance matrix', "lightblue", [-1.0, 1.0])
        else:    rightPlot = (right, 'sampled correlation matrix', "lightgreen", [-1.0, 1.0])
        x = arange(0, p, p / GRID)
    else:
        leftPlot = (left, 'elite marginal prob\'s', "yellow", [0., 1.])
        if diff: rightPlot = (left - right, 'deviance prob\'s', "lightblue", [-1., 1.])
        else: rightPlot = (right, ' sampled marginal prob\'s', "lightgreen", [0., 1.])

    # plot with rpy
    r.pdf(paper="a4r", width=12, height=12, file=PATH + "/steps/" + plotdir + "%i.pdf" % step)
    r.par(mfrow=[1, 2], oma=[0, 0, 2, 0], mar=[0, 0, 2, 0])
    for myPlot in [leftPlot, rightPlot]:
        if type == 'R':
            z = zeros((len(x), len(x)))
            for i in range(len(x)):
               for j in range(i - int(GRID / p)):
                   z[i][j] = myPlot[0][int(x[i])][int(x[j])]
            r.persp(x, x, z, box=True, xlab="", ylab="", zlab="", zlim=myPlot[3], theta=45, phi=25, col=myPlot[2])
        else:
            r.barplot(myPlot[0], axes=False, col=myPlot[2], ylim=myPlot[3])
        r.mtext("%i" % (step), outer=True, line=0, cex=2, col="red")
        r.title(main=myPlot[1])
    r.dev_off()

    
def plot4(post, m=1500, n=2000, verbose=True):
     '''
     Generates perfect samples from the posterior and initializes an
     independent, multinormal and log regression generator to plot their respective histograms.
     '''
     if not post.feasible:
         print "WARNING: Posterior of dimension %i not feasible for brute force exploration." % post.p
         return

     x = []; hist = array(4 * [zeros(2 ** post.p)])
   
     # explore posterior
     postdata = data()
     for d in range(2 ** post.p):
         b = dec2bin(d, post.p)
         x.append(bin2str(b))
         hist[0][d] = post.pmf(b)
         for j in range(int(m * hist[0][d])):
             postdata.append(b)
     hist[0] /= hist[0].sum()
        
     # init approximations
     gen = [binary_ind(postdata), binary_mn(postdata), binary_log(postdata)]
     for i in range(n):
         for g in range(3):
             d = bin2dec(gen[g].rvs())
             hist[g + 1][d] += 1.
     hist[[1, 2, 3]] /= n
     ymax = hist.max()

     # plot with rpy
     # "/home/cschafer/Documents/Python/workspace/data/rngs/all4_hist.pdf"
     r.pdf(paper="a4", file="/home/cschafer/Tex/amc_on_mbs/img/fragmaster/all4_hist.pdf", width=12, height=12)
     r.par(mfrow=[2, 2], oma=[40, 4, 0, 4], mar=[1, 2, 4, 1])
     for type in [\
         (hist[0], 'grey85', 'true mass function'), \
         (hist[1], 'grey45', 'independent model'), \
         (hist[2], 'grey45', 'latent normal model'), \
         (hist[3], 'grey45', 'logistic regression model')]:
         r.barplot(type[0], ylim=[0, ymax], names=x, cex_names=0.8, las=3, col=type[1], family="serif", cex_axis=0.8)
         r.title(main=type[2],line=1, family="serif", font_main=1, cex_main=1)
     #r.mtext("Histograms of n=%i based on m=%i pseudo samples" % (n, m), outer=True, line=1, cex=1.5)   
     #r.mtext("Histogram", outer=True, cex=1)   
     r.dev_off()

try:
    from rpy import *
    HASRPY = True
except:
    HASRPY = False
         
    def plot(self):
        '''
        Plots the probability distribution function of the generator (if possible).
        '''
        if not HASRPY: return
        x = []; y = zeros(2 ** self.p)
        
        for d in range(2 ** self.p):
            b = dec2bin(d, self.p)
            x.append(str(b))
            y[d] = self.prob(b)
    
        # normalize
        y = y / sum(y)

        # plot with R
        r.pdf(paper="a4r", file=self.name + "_plot.pdf", width=12, height=12, title="")
        r.barplot(y, names=x, cex_names=4. / self.p, las=3)
        r.title(main=self.type)


    def hist(self, n=10000):
        '''
        Plots a histogram of the empirical distribution obtained by sampling from the generator.
        '''
        if not hasrpy: return
        x = []; y = zeros(2 ** self.p)
                
        for i in range(n):
            b = self.rvs()
            y[bin2dec(b)] += 1.
        
        for d in range(2 ** self.p):
            b = dec2bin(d, self.p)
            x.append(str(b))
    
        # normalize
        y = y / sum(y)
        
        # plot with R
        r.pdf(paper="a4r", file=self.name + "_hist.pdf", width=12, height=12, title="")
        r.barplot(y, names=x, cex_names=4. / self.p, las=3, col="lightblue")
        r.title(main=self.type)
