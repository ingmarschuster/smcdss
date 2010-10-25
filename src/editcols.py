'''
USAGE: editcols [options] dataset [output]

    -h, --help
        display help message
                                
    -v, --variates
        variates as singles separated by commas (e.g. 2,4,7) or as a sequence (e.g. 2:7)

    -r, --response
        move that column to front

    -i, --intercept
        add a constant column
                                
    -c, --cross
        cross columns
        
REMARK: If the same variate is given several times or a binary variable is squared,
some Gaussian noise is automatically added to keep the columns distict.
    
'''

import csv, os, glob, sys, getopt, copy
from numpy import array, vstack, zeros, newaxis, empty, random, abs

def editcols(args):
    
    # prepare variate index
    if len(args['variates']) == 0:
        datreader = csv.reader(open(args['dataset'] + '_dat.csv'), delimiter=';'); row = datreader.next()
        args['variates'] = range(1, len(row) + 1)
    if args['intercept']:
        variates = [args['response']] + [0] + args['variates']
    else:
        variates = [args['response'] - 1] + list(array(args['variates']) - 1)
    
    # read doc
    docreader = csv.DictReader(open(args['dataset'] + '_doc.csv'), fieldnames=['column', 'type', 'short', 'long'], delimiter=';')
    column_names = [];
    if args['intercept']: column_names.append(dict([['column', '0'], ['type', 'd'], ['short', 'CONST'], ['long', '']]))
    for row in docreader:
        column_names.append(row)
    
    # add noise
    column_names = [copy.copy(col) for col in array(column_names)[variates]]
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
    datreader = csv.reader(open(args['dataset'] + '_dat.csv'), delimiter=';'); data = []
    for row in datreader:
        if len(row) == 0: break
        if args['intercept']: row.insert(0, 1.0)
        x = empty(1 + d, dtype=float);
        for index, entry in enumerate(array(row)[variates]):
            x[index] = float(entry)
            if column_names[index]['short'][-7:] == '+noise)':
                x[index] = x[index] + random.normal(0, 0.1)
        data.append(x)
    n = len(data)
    data = array(data)
    
    # cross columns
    intercept = int(args['intercept'])
    if args['cross']:
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
    writer = csv.writer(open(args['output'] + '_doc.csv', 'wb'), delimiter=';')
    for column in column_names:
        writer.writerow([column['column'], column['type'], column['short'], column['long']])
    
    # write data
    writer = csv.writer(open(args['output'] + '_dat.csv', 'wb'), delimiter=';')
    for row in data:
        writer.writerow(list(row))
   
    
def main():
    
    # dict with arguments
    args = dict([\
        ['response', 1], \
        ['variates', []], \
        ['dataset', ''], \
        ['output', ''], \
        ['intercept', False], \
        ['cross', False], \
        ])
    
    # parse command line arguments
    try:
        opts, lineargs = getopt.getopt(sys.argv[1:], "r:v:ich", \
        ['response=', 'variates=', 'intercept', 'cross', 'help'])
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
        if o in ("-r", "--response"):
            args['response'] = int(a)
        if o in ("-v", "--variates"):
            args['output'] = a.replace(',', '.').replace(':', '-')
            try:
                p = a.find(':')
                if p > 0:
                    args['variates'] = range(int(a[0:p]), int(a[p + 1:]) + 1)
                else:
                    args['variates'] = eval('[' + a + ']')
            except:
                print "Error while parsing variates " + a + "."
        if o in ("-c", "--cross"):
            args['cross'] = True
        if o in ("-i", "--intercept"):
            args['intercept'] = True

    args['dataset'] = lineargs[0]
    if args['dataset'][-8:] == '_dat.csv':
        args['dataset'] = args['dataset'][:-8]
    if len(lineargs) > 1:
        args['output'] = lineargs[1]
    else:
        args['output'] = args['dataset'] + '_' + args['output']

    editcols(args)
 
if __name__ == "__main__":
    main()
