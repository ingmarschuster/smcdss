#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Logging stdout into log files. @namespace utils.logger """

import time
import numpy

class Logger:
    """
        Capture print statments and write them to a log file
        but still allow them to be printed on the screen.
    """
    def __init__(self, stdout, filename):
        filename += '%d.txt' % numpy.random.randint(low=0, high=10e8)

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
