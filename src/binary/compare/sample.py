import os
import numpy

from binary.selector_glm import w_logistic, w_probit

def main():

    path = os.path.expanduser('~/Documents/Data/bvs/link')
    n = 500
    d = 50
    noise = 0.0

    filename = 'logit_%.1f.csv' % noise

    link = w_logistic

    beta = numpy.hstack((numpy.linspace( .8, .2, 10),
                         numpy.linspace(-.8,-.2, 10)))
    p = beta.shape[0]
    print repr(beta)
    beta_0 = 0.0

    X = numpy.random.standard_normal(size=(n, d))
    y = numpy.zeros(n, dtype=int)
    for k in xrange(n):
        prob = link(beta_0 + numpy.dot(X[k, :p], beta) + noise * numpy.random.standard_normal())
        if numpy.random.random() < prob:
            y[k] = 1

    f = open(os.path.join(path, filename), 'w')
    f.write(','.join(['y'] + ['x%d' % (i + 1) for i in xrange(d)]) + '\n')
    for k in xrange(n):
        f.write(','.join(['%d' % y[k]] + ['%.6f' % x for x in X[k]]) + '\n')
    f.close()

if __name__ == "__main__":
    main()
