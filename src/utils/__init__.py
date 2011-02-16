import os, sys
import numpy as np
import pyximport

import python
opts = ['python']

if os.name == 'nt':
    if os.environ.has_key('CPATH'):
        os.environ['CPATH'] = os.environ['CPATH'] + np.get_include()
    else:
        os.environ['CPATH'] = np.get_include()
    
    mingw_setup_args = { 'options': { 'build_ext': { 'compiler': 'mingw32' } } }
    pyximport.install(setup_args=mingw_setup_args)

if os.name == 'posix':
    pyximport.install()
    try:
        import weave
        opts += ['weave']
    except Exception, exception:
        sys.stderr.write('weave exception: ' + str(exception) + '\n') 
        
if True:
    import cython
    opts += ['cython']
#except:
#    print "cython error:", sys.exc_info()[0]