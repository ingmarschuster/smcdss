#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
    
'''

import time, os
import numpy

class Logger:
    '''
        Capture print statments and write them to a log file
        but still allow them to be printed on the screen.
    '''
    def __init__(self, stdout, filename):
        for i in xrange(1, 100):
            if not os.path.isfile(filename + '%0*d.txt' % (2, i)):
                filename += '%0*d.txt' % (2, i)
                break

        self.stdout = stdout
        self.logfile = file(filename, 'w')
        self.logfile.write('\n\nLoggig run at %s\n\n' % time.ctime())

    def write(self, text):
        self.stdout.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        self.stdout.flush()

    def flush(self):
        self.stdout.flush()

    def close(self):
        self.stdout.close()
        self.logfile.close()
