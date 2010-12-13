#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    @author Christian Schäfer
    $Date$
    $Revision$
'''

__version__ = "$Revision$"

import csv, copy
from numpy import *

def editcols(param):
    '''
        Creates a data set with altered columns from a given data set.
        @see main
        @param dataset data set
        @param outfile name of created data set
        @param param dictionary of parameters
    '''

    # parse parameters
    response = param['data_regressand']
    variables = param['data_regressors']
    intercept = param['data_intercept']

    cross = param['data_cross']
    dataset = param['sys_path'] + '/' + param['data_path'] + '/' + param['data_set']
    outfile = param['sys_path'] + '/' + param['test_path'] + '/' + param['test_name'] + '/' + param['data_set']

    if intercept: variables = [response] + [0] + variables
    else: variables = [response - 1] + [i - 1 for i in variables]

    # read doc
    docreader = csv.DictReader(open(dataset + '_doc.csv'), fieldnames=['column', 'type', 'short', 'long'], delimiter=';')
    column_names = [];
    if intercept: column_names.append(dict([['column', '0'], ['type', 'd'], ['short', 'CONST'], ['long', '']]))
    for row in docreader:
        column_names.append(row)

    # add noise
    column_names = [column_names[i] for i in variables]
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
    if cross:
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

    return outfile + '_dat.csv'