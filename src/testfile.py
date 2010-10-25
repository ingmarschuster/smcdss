#
# This is to test whether the approximate versions of the conditional
# of the log-linear model is good enough for sampling from the model.
#
# A note on the quadratic exponential binary distribution
# Author(s): D.R. Cox, Nanny Wermuth
# Source: Biometrika 1994, 81, 2, pp. 403-8
#

from numpy import *

# Converts an integer into a +1,-1 array containing its binary representation.
def dec2bin(n, dim=0):
    b = []
    while n > 0:
        if n % 2:
            b.append(1)
        else:
            b.append(-1)            
        n = n >> 1
    while len(b) < dim:
        b.append(-1)
    b.reverse()
    return array(b)

# Generates a random log-linear model.
def randomModel(dim):
    U = random.normal(0, .25, dim * dim).reshape((dim, dim))
    alpha = dot(U, U.T)
    alpha = dot(alpha, diag(2 * random.binomial(1, .5, 6) - ones(DIM)))
    return alpha

# Evaluates the log-linear model.
def loglinear(b):
    sum = dot(diag(ALPHA), b)
    for i in range(DIM):
        for j in range(i):
            sum += b[i] * b[j] * ALPHA[i, j]
    return exp(MU + sum)

# Compute the normalization constant.
def computeMU():
    sum = 0
    for dec in range(2 ** DIM):
        b = dec2bin(dec, DIM)
        sum += loglinear(b)
    return - log(sum)



# dimension
DIM = 6

# a log-linear model
ALPHA= array(\
[-0.97882863, 0.61502143, 0.03769291, 0.23966928, 0.31955478, 0.02240849,\
  0.61502143,-0.58855963,-0.08270103,-0.29593133,-0.13480816,-0.17921533,\
 -0.03769291, 0.08270103, 0.17368193, 0.08290546,-0.04733021, 0.15170509,\
  0.23966928,-0.29593133,-0.08290546,-0.49613294, 0.26298699,-0.32414626,\
  0.31955478,-0.13480816, 0.04733021,-0.26298699,-0.35536491,-0.01111059,\
 -0.02240849, 0.17921533, 0.15170509, 0.32414626, 0.01111059, 0.35519851]).reshape((DIM,DIM))

# normalization constant
MU = 0
MU = computeMU()


# Computes the log-linear approximately marginalized over the dim - len(b) components.
def marginal_loglinear(b):
    B = range(len(b))
    C = range(len(b), DIM)
    
    mu = MU + len(C) * log(2)
    for r in C:
        mu += log(cosh(ALPHA[r, r]))
        for j in B:
              mu += .5 * ALPHA[j, r] ** 2 * cosh(ALPHA[r, r]) ** -2

    alpha = copy(ALPHA)[:len(B), :len(B)]
    for j in B:
        for r in C:
            alpha[j, j] += ALPHA[j, r] * tanh(ALPHA[r, r]) 
            for s in C:
                  if r > s:
                      alpha[j, j] += ALPHA[j, r] * ALPHA[r, s] * tanh(ALPHA[s, s]) * cosh(ALPHA[r, r]) ** -2
    for j in B:
        for k in B:
            if j > k:
                for r in C:
                    alpha[j, k] += ALPHA[j, r] * ALPHA[k, r] * cosh(ALPHA[r, r]) ** -2
    
    sum = dot(diag(alpha), b)
    for i in B:
        for j in range(i):
            sum += b[i] * b[j] * alpha[i, j]
    return exp(MU + sum)

# Computes the first moments.
def computeMOMENTS(dim):
    m0 = 0
    m1 = zeros(dim)
    m2 = zeros((dim, dim))
    for dec in range(2 ** dim):
        b = dec2bin(dec, dim)
        p = marginal_loglinear(b)
        m0 += p
        m1 += b * p
        m2 += dot(b[:, newaxis], b[newaxis, :]) * p
    cov = m2 - dot(m1[:, newaxis], m1[newaxis, :])
    var = diag(cov)
    cov /= sqrt(dot(var[:, newaxis], var[newaxis, :]))
    return m0, m1, cov

# Displays moments.
def printMOMENTS(dim):
    MOMENTS = computeMOMENTS(dim)
    
    print MOMENTS[0]
    print
    print ' '.join('% 5f' % n for n in MOMENTS[1])
    print
    for i in range(dim):
        print ' '.join('% 5f' % n for n in MOMENTS[2][i, :])


printMOMENTS(5)
