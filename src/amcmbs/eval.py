'''
USAGE: eval [OPTIONS]

    -h, --help
                                display help message
                                
    -f, --files       =file.txt(,file.txt)
                                evaluate result file, create file.eval.txt and file.pdf 

    -c, --color
                                make colored graph
                                
    -o, --okular
                                open all pdfs in tex/img folder
                                       
    -t, --titles
                                include generic titles   

    -n, --number
                                number of modes to include (for opt)

    -b, --boxed       =float
                                percentage of data in center box
'''

from numpy import *
import csv, os, glob, sys, getopt, platform
try:
    from plotting import *
except:
    print 'Can\'t import rpy for plotting.'

TEST_PATH = '/home/cschafer/Documents/Python/smcdss/data/testruns'
PDF_PATH = '/home/cschafer/Documents/Python/smcdss/data/testruns'

def maketitle(file):
    file = PDF_PATH + "/" + file
    if not os.path.isfile(file + "_fm"):
        os.system ("cp %s %s" % (PDF_PATH + "/generic", file + "_fm"))
    os.system ("pdftops -eps " + file + ".pdf")    
    os.system ("mv " + file + ".eps" + " " + file + "_fm.eps")
    os.system ("rm " + file + ".pdf")    
    os.system (PDF_PATH + "/fragmaster.pl > /dev/null")
    os.system ("mv " + file + ".pdf " + file.replace('fragmaster/', '') + ".pdf")   

def evalest(filestubs=['*'], boxdata=0.8, titles=False, color=False):
    '''
    Make pdf-boxplots from testfiles.
    
    @param file Name of the testfile to be processed.
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
        files = glob.glob(TEST_PATH + '/' + filestub + '_*.txt')
        if len(files) == 0:
            print "No results files in " + filestub + "_*.txt."
            exit()

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
    r.pdf(paper="a4r", file=PDF_PATH + "/" + ''.join(filestubs) + ".pdf", width=12, height=12)
    if color:
        colors = ['azure1', 'black', 'white', 'white', 'black']
    else:
        colors = ['grey85', 'black', 'white', 'white', 'black']
    
    #if titles:
        #r.par(mfcol=[len(filestubs), 1], oma=[30, 4, 5 , 4], mar=[2, 0, 2, 0])
    #    r.par(mfrow=[len(filestubs), 1], oma=[0, 0, 1 , 0], mar=[2, 0, 2, 0])
    #else:
    r.par(mfcol=[len(filestubs), 1], oma=[2, 2, 2 , 2], mar=[0, 0, 2, 0])        
    for ifilestub in range(len(filestubs)):
        if titles:
            title = "title " + str(ifilestub)
        else:
            title = filestubs[ifilestub] + ": ths. evals=%.1f, time=%.3f" % \
                (evaldict['targetevals'][ifilestub], evaldict['time'][ifilestub]) + \
                ", " + str("boxdata=%i%%, test runs=%i," % (boxdata * 100, evaldict['n'][ifilestub])) + evaldict['header'][ifilestub]
        r.barplot(evaldict['quantiles'][ifilestub], ylim=[0, 1], axes=False, col=colors)
        r.title(main=title, line=1, family="serif", cex_main=1, font_main=1)
    r.dev_off()
    if titles:
        maketitle(file=''.join(filestubs))
 

def evalopt(filestubs=['*'], number=None, boxdata=0.8, titles=False, color=False):
       
    evaldict = dict([\
        ['header', ['', '']], \
        ['hist', [[], []]], \
        ['targetevals', array([0., 0.])], \
        ['time', array([0., 0.])], \
        ['mode', [[], []]], \
        ['score', [[], []]], \
        ])

    for ifilestub, filestub in enumerate(filestubs):       
        files = glob.glob(TEST_PATH + '/' + filestub + '_*.txt')
        if len(files) == 0:
            print "No results files in " + filestub + "_*.txt."
            exit()
        
        datreader = csv.reader(open(files[0]), delimiter=';')
        evaldict['header'] = datreader.next()[0];
        datreader.next()
        
        print "Processing " + filestub + "..." 
        for file in files:
            datreader = csv.reader(open(file), delimiter=';')
            datreader.next(); datreader.next()
            for row in datreader:
                if len(row) == 0: continue
                score = float(row[0])
                mode = eval(row[1])
                if not mode in evaldict['mode'][ifilestub]:
                    evaldict['score'][ifilestub].append(score);
                    evaldict['hist'][ifilestub].append(0)
                    evaldict['mode'][ifilestub].append(eval(row[1]))
                evaldict['hist'][ifilestub][evaldict['mode'][ifilestub].index(mode)] += 1
                evaldict['time'][ifilestub] += float(row[2])
                evaldict['targetevals'][ifilestub] += float(row[3])

    if len(filestubs) > 1:               
        for imode, mode in enumerate(evaldict['mode'][0]):
            if not mode in evaldict['mode'][1]:
                evaldict['score'][1].append(evaldict['score'][0][imode])
                evaldict['hist'][1].append(0)
                evaldict['mode'][1].append(mode)
            else:
                evaldict['score'][1][evaldict['mode'][1].index(mode)] = evaldict['score'][0][imode]
        for imode, mode in enumerate(evaldict['mode'][1]):
            if not mode in evaldict['mode'][0]:
                evaldict['score'][0].append(evaldict['score'][1][imode])
                evaldict['hist'][0].append(0)
                evaldict['mode'][0].append(mode)
   
    if number == None: number = len(evaldict['mode'][0])
    
    for ifilestub, filestub in enumerate(filestubs):
        evaldict['score'][ifilestub] = array(evaldict['score'][ifilestub])
        index = evaldict['score'][ifilestub].argsort()[::-1] # order ascending
        evaldict['score'][ifilestub] = evaldict['score'][ifilestub][index]
        evaldict['mode'][ifilestub] = [evaldict['mode'][ifilestub][i] for i in index]
        evaldict['hist'][ifilestub] = array(evaldict['hist'][ifilestub])[index]
        n = evaldict['hist'][ifilestub].sum()
        evaldict['time'][ifilestub] /= float(n)
        evaldict['targetevals'][ifilestub] /= float(n)
        # resume numbers
        evaldict['hist'][ifilestub][number - 1] = evaldict['hist'][ifilestub][number - 1:].sum()
        evaldict['hist'][ifilestub] = evaldict['hist'][ifilestub][0:number]
    
    ylim = int(evaldict['hist'][0].max())
    if len(filestubs) > 1: ylim = int(max(ylim, evaldict['hist'][1].max()))

    # list models and corresponding scores
    evaltxt = "score and model\n"
    for ix in range(number):
        evaltxt += ("%i" % ix).rjust(3) + " : "
        evaltxt += ("%i" % evaldict['hist'][0][ix]).rjust(3) + " : "
        evaltxt += ("%i" % evaldict['hist'][1][ix]).rjust(3) + " : "
        evaltxt += "%.8f : " % evaldict['score'][0][ix]
        evaltxt += str(evaldict['mode'][0][ix]) + "\n"

    # compute model overlap
    evaltxt += "\nmodel overlap:\n    " + ''.join([("%3i " % i) for i in range(1, number + 1)]) + "\n"
    cont = zeros((number, number))
    for ix in range(number):
        for varx in evaldict['mode'][0][ix]:
            for iy in range(number):
                for vary in evaldict['mode'][0][iy]:
                    if varx == vary:
                        cont[ix][iy] += 1
        evaltxt += ("%3i " % (ix + 1)) + ''.join([("%3i " % i) for i in (cont[ix, :])]) + "\n"
    
    # compute relative model overlap
    evaltxt += "\nrelative overlap:\n    " + ''.join([("%3i " % i) for i in range(1, number + 1)]) + "\n"
    cont /= sqrt(dot(cont.diagonal()[:, newaxis], cont.diagonal()[newaxis, :]))
    for ix in range(number):
        evaltxt += ("%3i " % (ix + 1)) + ''.join([("%3i " % i) for i in 100 * cont[ix, :]]) + "\n"

    # extract files
    file = open(TEST_PATH + "/" + ''.join(filestubs) + ".eval", 'w')
    file.write(evaltxt)
    file.close()

    # plot with rpy
    r.pdf(paper="a4", file=PDF_PATH + "/" + ''.join(filestubs) + ".pdf", width=12, height=12)
    if color: colors = ['yellow', 'lightgreen']
    else: colors = ['grey85', 'grey45']
    r.par(mfrow=[1, len(filestubs)], oma=[40, 4, 5, 4], mar=[0, 0, 0, 0], family="serif")
    title_lineskips = [2, 2] # controls height of title
    for ifilestub, stub in enumerate(filestubs):
        if titles:
            title = 'title ' + str(ifilestub)
        else:
            title = stub + ": time %.3f, targetevals %.3f" % (evaldict['time'][ifilestub], evaldict['targetevals'][ifilestub])
        r.barplot(evaldict['hist'][ifilestub], ylim=[0, ylim], axes=False, names=range(1, number) + ['out'], \
                  cex_names=2 / sqrt(number + 1), las=1, col=colors[ifilestub])
        r.title(main=title, line= -title_lineskips[ifilestub], cex_main=1, font_main=1)

    if not titles:
        r.mtext(str("test runs=%i/" % n) + evaldict['header'], outer=True, line=1, cex=1)
    r.dev_off()
        
    if titles:
        maketitle(file=''.join(filestubs))

def main():
    '''
    Parse command line options to smctest. Type evalt --help for help.
    '''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "cf:thb:n:o", ['color', 'number=', 'boxdata=', 'okular', 'help', 'files=', 'titles'])
    except getopt.error, msg:
        print msg
        sys.exit(2)
    
    # Set default parameters.
    boxdata = 0.8
    titles = False
    filestubs = ['*']
    color = False
    number = None

    # Process options.
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        if o in ("-t", "--titles"): titles = True
        if o in ("-n", "--number"): number = int(a)
        if o in ("-o", "--okular"): exit()
        if o in ("-c", "--color"): color = True
        if o in ("-b", "--boxed"): boxdata = float(a)
        if o in ("-f", "--files"): filestubs = eval("['" + a.replace(",", "','") + "']")

    if filestubs[0].find('ceopt') > -1 or filestubs[0].find('saopt') > -1:
        evalopt(filestubs=filestubs, number=number, boxdata=boxdata, titles=titles, color=color)
    else:
        evalest(filestubs=filestubs, boxdata=boxdata, titles=titles, color=color)        

if __name__ == "__main__":
    main()
