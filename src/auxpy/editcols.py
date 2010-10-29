#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

import sys
import csv
from getopt import getopt
from copy import copy
from numpy import array, vstack, zeros, newaxis, empty, random, abs

def editcols(dataset, outfile, **param):
    '''
        Creates a data set with altered columns from a given data set.
        @see main
        @param dataset data set
        @param outfile name of created data set
        @param param dictionary of parameters
    '''

    # parse parameters
    response = 1
    if 'response' in param:
        response = param['response']

    if 'variables' in param:
        variables = param['variables']
    else:
        datreader = csv.reader(open(dataset + '_dat.csv'), delimiter=';'); row = datreader.next()
        variables = range(1, len(row) + 1)
    
    intercept = ('intercept' in param)
    if intercept:
        variables = [response] + [0] + variables
    else:
        variables = [response - 1] + list(array(variables) - 1)

    # read doc
    docreader = csv.DictReader(open(dataset + '_doc.csv'), fieldnames=['column', 'type', 'short', 'long'], delimiter=';')
    column_names = [];
    if intercept: column_names.append(dict([['column', '0'], ['type', 'd'], ['short', 'CONST'], ['long', '']]))
    for row in docreader:
        column_names.append(row)

    # add noise
    column_names = [copy(col) for col in array(column_names)[variables]]
    d = len(column_names) - 1
    for i in range(1, d):
        x = column_names[i]['column']
        if x.find('+') > -1:
            x = x[: x.find('+')]
        if x == column_names[i + 1]['column']:
            column_names[i + 1]['short'] = column_names[i]['short'] + '+noise'; column_names[i + 1]['column'] = column_names[i]['column'] + '+noise'

    # move short to long, column to short, add parenthesis
    for i in range(1 + d):
        column_names[i]['long'] = column_names[i]['short']; column_names[i]['short'] = column_names[i]['column']
        x = column_names[i]['short']
        if x.find('+') > -1:
            column_names[i]['long'] = '(' + column_names[i]['long'] + ')'; column_names[i]['short'] = '(' + column_names[i]['short'] + ')'

    # read data
    datreader = csv.reader(open(dataset + '_dat.csv'), delimiter=';'); data = []
    for row in datreader:
        if len(row) == 0: break
        if intercept: row.insert(0, 1.0)
        x = empty(1 + d, dtype=float);
        for index, entry in enumerate(array(row)[variables]):
            x[index] = float(entry)
            if column_names[index]['short'][-7:] == '+noise)':
                x[index] = x[index] + random.normal(0, 0.1)
        data.append(x)
    n = len(data)
    data = array(data)

    # cross columns
    intercept = int(intercept)
    if 'cross' in param:
        data = vstack((data.T, empty((n, d * (d + 1 - 2 * intercept) / 2), dtype=float).T)).T
        column_names_crossed = []
        for i in range(d + 1 - intercept):
            for j in range(i):

                # cross data
                data[:, 1 + d + j + i * (i - 1) / 2] = data[:, i + intercept] * data[:, j + intercept + 1]

                # set data type
                column_dict = dict([['column', str(1 + d + j + i * (i - 1) / 2)], \
                                             ['short', column_names[i + intercept]['short'] + '*' + column_names[j + intercept + 1]['short']], \
                                             ['type', 'd'], \
                                             ['long', column_names[i + intercept]['long'] + '*' + column_names[j + intercept + 1]['long']]])
                if column_names[i + intercept]['type'] == 'c' or column_names[j + intercept + 1]['type'] == 'c': column_dict['type'] = 'c'

                # add noise to squared binary column
                if column_names[i + intercept]['type'] == 'b' and column_names[j + intercept + 1]['type'] == 'b':
                    column_dict['type'] = 'c'
                    column_dict['short'] = column_dict['short'] + '+noise'
                    column_dict['long'] = column_dict['long'] + '+noise'
                    data[:, 1 + d + j + i * (i - 1) / 2] = data[:, 1 + d + j + i * (i - 1) / 2] + random.normal(0, 0.1, n)

                column_names_crossed.append(column_dict)
        column_names = column_names + column_names_crossed
        d = d * (d + 1) / 2

    # enumerate columns
    for i in range(d + 1):
        column_names[i]['column'] = str(i)

    # write doc
    writer = csv.writer(open(outfile + '_doc.csv', 'wb'), delimiter=';')
    for column in column_names:
        writer.writerow([column['column'], column['type'], column['short'], column['long']])

    # write data
    writer = csv.writer(open(outfile + '_dat.csv', 'wb'), delimiter=';')
    for row in data:
        writer.writerow(list(row))


def main():
    '''
    USAGE: editcols [options] dataset [output]
    
    -h, --help
        display help message
                                
    -v, --variables
        variables as singles separated by commas (e.g. 2,4,7) or as a sequence (e.g. 2:7)
    
    -r, --response
        move that column to front
    
    -i, --intercept
        add a constant column
                                
    -c, --cross
        cross columns
    '''

    # parse command line arguments
    try:
        opts, args = getopt(sys.argv[1:], "r:v:ich",
                                ['response=', 'variables=', 'intercept', 'cross', 'help'])
    except getopt.error, msg:
        print msg
        sys.exit(2)

    if len(args) == 0:
        print main.__doc__
        sys.exit(2)

    # parse command line options
    param = dict()
    for o, a in opts:
        if o in ("-h", "--help"):
            print main.__doc__
            sys.exit(0)

        if o in ("-r", "--response"):
            param['response'] = int(a)

        if o in ("-v", "--variables"):
            outfile = a.replace(',', '.').replace(':', '-')
            try:
                a = a.split(':')
                if len(a) == 1:
                    param['variables'] = eval('[' + a[0] + ']')
                else:
                    param['variables'] = range(int(a[0]), int(a[1]) + 1)
            except:
                print "Error while parsing variables " + a + "."

        if o in ("-c", "--cross"): param['cross'] = True

        if o in ("-i", "--intercept"): param['intercept'] = True

    dataset = args[0]
    if dataset[-8:] == '_dat.csv': dataset = dataset[:-8]

    # check outfile
    if len(args) > 1:
        outfile = args[1]
        # add path of dataset
        if outfile.find('/') == -1:
            outfile = dataset[:dataset.rfind('/') + 1] + outfile
    else:
        outfile = ''.join(dataset, '_', outfile)

    editcols(dataset, outfile, **param)

if __name__ == "__main__":
    main()
