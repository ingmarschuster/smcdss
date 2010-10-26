'''
Created on 19 oct. 2010

@author: cschafer
'''

def bin2dec(b):
    '''
    Converts a boolean array into an integer.
    '''
    return long(bin2str(b), 2)

def bin2str(b):
    '''
    Converts a boolean array into a binary string.
    '''
    return ''.join([str(i) for i in array(b, dtype=int)])

def dec2bin(n, dim=0):
    '''
    Converts an integer into a boolean array containing its binary representation.
    '''
    b = []
    while n > 0:
        if n % 2:
            b.append(True)
        else:
            b.append(False)            
        n = n >> 1
    while len(b) < dim:
        b.append(False)
    b.reverse()
    return array(b)
