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
    def __init__(self, stdout, filename=None, textfield=None):

        self.filename = filename
        if not self.filename is None:
            self.filename += '%d.txt' % numpy.random.randint(low=0, high=10e8)
            self.logfile = file(filename, 'w')
            self.logfile.write('\n\nLoggig run at %s\n\n' % time.ctime())

        self.textfield = textfield

        self.stdout = stdout

    def write(self, text):
        self.stdout.write(text)
        self.stdout.flush()
        if not self.filename is None:
            self.logfile.write(text)
            self.logfile.flush()
        if not self.textfield is None:
            self.textfield.write(text)

    def flush(self):
        self.stdout.flush()

    def close(self):
        self.stdout.close()
        if not self.filename is None:
            self.logfile.close()
