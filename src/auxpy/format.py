'''
@author: cschafer
'''

def format_vector(v):
    ''' Formats a vector for output on stdout '''
    return '[' + ' '.join([('%.4f' % x).rjust(7) for x in v]) + ' ]\n'

def format_matrix(M):
    ''' Formats a matrix for output on stdout '''
    return ''.join([format_vector(x) for x in M])