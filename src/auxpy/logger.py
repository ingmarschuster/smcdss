"""
Capture print statments and write them to a log file
but still allow them to be printed on the screen.
Usage:  see if __name__=='__main__': section below.
"""

import sys
import time, datetime
from numpy.random import randint

class Logger:
    def __init__(self, stdout, filename):
        filename += '%06i.txt' % randint(1e6)
        self.stdout = stdout
        self.logfile = file(filename, 'w')
        self.logfile.write('\n\nLoggig run at %s\n\n' % time.ctime())
    def write(self, text):
        self.stdout.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        self.stdout.flush()
    def close(self):
        self.stdout.close()
        self.logfile.close()
