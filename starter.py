'''
USAGE: editcols [options] inifile [output]

    -h, --help
        display help message
                                
    -r, --run
        run option: MCMC, SMC, CE or SA
  
'''

import getopt, sys
import ConfigParser

def starter(args):
    cp = ConfigParser.RawConfigParser()
    cp.read("default.ini")
    #cp.read("/home/cschafer/Documents/Python/workspace/data/datasets/test.ini")
    print cp.sections()
    print cp.options('eval')
    print eval(cp.get('eval', 'outer_margin'))
    print cp.get('eval', 'title_line')

def main():
    
    # dict with arguments
    args = dict([\
        ['run', 'mcmc'], \
        ['inifile', ''], \
        ['output', ''], \
        ])
    
    # parse command line arguments
    try:
        opts, lineargs = getopt.getopt(sys.argv[1:], "hr:", ['help', 'run='])
    except getopt.error, msg:
        print msg
        sys.exit(2)
        
    if len(lineargs) == 0:
        print __doc__
        sys.exit(2)

    # parse command line options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        if o in ("-r", "--run"):
            args['run'] = a

    args['inifile'] = lineargs[0]
    if len(lineargs) > 1:
        args['output'] = lineargs[1]
    else:
        args['output'] = args['inifile'][:-4] + '_out'

    starter(args)
 
if __name__ == "__main__":
    main()
