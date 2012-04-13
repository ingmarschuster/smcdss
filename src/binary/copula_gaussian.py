#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family obtained via dichotomizing a multivariate Gaussian.
    \namespace binary.gaussian_copula
    \details The correlation structure of the model is limited by the constraints of the elliptic Gaussian copula.
"""

import binary.copula as copula
import binary.wrapper as wrapper

import numpy
import scipy.stats as stats

class GaussianCopulaBinary(copula.CopulaBinary):
    """ Binary parametric family obtained via dichotomizing a multivariate Gaussian. """

    NU = 1.0

    def __init__(self, p, R, delta=None, name='Gaussian copula family', long_name=__doc__):
        """ 
            Constructor.
            \param p mean
            \param R correlation matrix
        """

        # call super constructor
        super(GaussianCopulaBinary, self).__init__(p, R, name=name, long_name=long_name)

        self.py_wrapper = wrapper.copula_gaussian()

        # add modules
        self.pp_modules += ('binary.copula_gaussian',)

    @classmethod
    def _rvs(cls, V, mu, C):
        """ 
            Generates a random variable.
            \param V normal variables
            \param param parameters
            \return binary variables
        """
        Y = numpy.empty((V.shape[0], V.shape[1]), dtype=bool)
        for k in xrange(V.shape[0]):
            Y[k] = mu > numpy.dot(C, V[k])
        return Y

    def _rvsbase(self, size):
        return numpy.random.normal(size=(size, self.d))

    @classmethod
    def newtonraphson(cls, mu, p, r, init=0, verbose=False):
        """
            Newton-Raphson search for the correlation parameter q of the underlying normal distribution.
            \param mu mean of the hidden stats.normal
            \param p mean of the binary            
            \param r correlation between the binary
            \param init initial value
            \param verbose print to stdout 
        """
        if verbose: print '\nNewton-Raphson search.'

        t = numpy.sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
        s = p[0] * p[1] + r * t

        greater_one = False
        q = init
        last_q = numpy.inf

        for i in xrange(GaussianCopulaBinary.MAX_ITERATIONS):
            try:
                q = q - round((bvnorm.cdf(mu, q) - s), 8) / bvnorm.pdf(mu, q)
            except FloatingPointError:
                return 0.0, i
            if verbose: print q
            if q > 1:
                q = 0.999
                if greater_one == True:
                    break                  # avoid endless loop
                else:
                    greater_one = True     # restart once at boundary

            if q < -1:
                break

            if abs(last_q - q) < GaussianCopulaBinary.PRECISION: break
            last_q = q

        return q, i

    @classmethod
    def aux_cdf(cls, x, r):
        return bvnorm.cdf(x, r)

    @classmethod
    def aux_ppf(cls, p):
        return stats.norm.ppf(p)


#-------------------------------------------------------------- bivariate normal

class _bvnorm(stats.rv_continuous):
    """
        bivariate normal distribution with correlation coefficient r.
        pdf(x,y) = numpy.exp(-(x*x-2*r*x*y+y*y)/(2*(1-r*r))) / (2*numpy.pi*numpy.sqrt(1-r*r))
    """

    def cdf(self, x, r=0):
        """
            Computes the bivariate normal cumulative distribution function,
            i.e. the probability that X < x and Y < y. The function only calls lowerDW(x, y, r).
            \param x value
            \param r correlation coefficient 
        """
        return self.lower_DW(x[0], x[1], r)

    def pdf(self, x, r=0):
        """
            Computes the bivariate normal probability distribution function, i.e. the density at (x, y)
            \param x value
            \param r correlation coefficient 
        """
        z = x[0] * x[0] - 2 * r * x[0] * x[1] + x[1] * x[1]
        return numpy.exp(-z / (2 * (1 - r * r))) / (2 * numpy.pi * numpy.sqrt(1 - r * r))

    def rvs(self, r=0):
        """
            \param r correlation coefficient 
            \return random bivariate normal
        """
        v = numpy.random.normal(0, 1)
        return numpy.array([v, r * v + numpy.sqrt(1 - r * r) * numpy.random.normal(0, 1)])

    def lower_DW(self, dh, dk, r):
        """
            Computes bivariate normal probabilities; lowerDW calculates the probability
            that x < dh and y < dk using the Drezner-Wesolowsky approximation.
            The function only calls upperDW(-dh, -dk, r).
            
            \param dh 1st lower integration limit
            \param dk 2nd lower integration limit
            \param r correlation coefficient
        """
        return self.upper_DW(-dh, -dk, r)

    def upper_DW(self, dh, dk, r):
        """
            Computes bivariate normal probabilities; upperDW calculates the probability that x > dh and y > dk. 
              
            This function is based on the method described by Z. Drezner and G.O. Wesolowsky, (1989),
            "On the computation of the bivariate normal integral", Journal of Statist.
            Comput. Simul. 35, pp. 101-107, with major modifications for double precision, for |r| close to 1.
        
            The code was adapted for python from the matlab routine by Alan Genz.
            
            \param dh 1st lower integration limit
            \param dk 2nd lower integration limit
            \param r correlation coefficient
        """
        twopi = 2 * numpy.pi
        if abs(r) < 0.3:
            lg = 3
        #       Gauss Legendre points and weights, n =  6
            w = [0.1713244923791705, 0.3607615730481384, 0.4679139345726904]
            x = [0.9324695142031522, 0.6612093864662647, 0.2386191860831970]
        elif abs(r) < 0.75:
            lg = 6;
        #       Gauss Legendre points and weights, n = 12
            w = [.04717533638651177, 0.1069393259953183, 0.1600783285433464, \
                 0.2031674267230659, 0.2334925365383547, 0.2491470458134029]
            x = [0.9815606342467191, 0.9041172563704750, 0.7699026741943050, \
                 0.5873179542866171, 0.3678314989981802, 0.1252334085114692]
        else:
            lg = 10
        #       Gauss Legendre points and weights, n = 20
            w = [.01761400713915212, .04060142980038694, .06267204833410906, \
                       .08327674157670475, 0.1019301198172404, 0.1181945319615184, \
                       0.1316886384491766, 0.1420961093183821, 0.1491729864726037, \
                       0.1527533871307259]
            x = [0.9931285991850949, 0.9639719272779138, 0.9122344282513259, \
                       0.8391169718222188, 0.7463319064601508, 0.6360536807265150, \
                       0.5108670019508271, 0.3737060887154196, 0.2277858511416451, \
                       0.07652652113349733]

        h = dh; k = dk; hk = h * k; bvn = 0
        if abs(r) < 0.925:
            hs = (h * h + k * k) / 2; asr = numpy.arcsin(r);
            for i in range(lg):
                sn = numpy.sin(asr * (1 - x[i]) / 2);
                bvn = bvn + w[i] * numpy.exp((sn * hk - hs) / (1 - sn * sn));
                sn = numpy.sin(asr * (1 + x[i]) / 2);
                bvn = bvn + w[i] * numpy.exp((sn * hk - hs) / (1 - sn * sn));
            bvn = bvn * asr / (4 * numpy.pi) + stats.norm.cdf(-h) * stats.norm.cdf(-k)
        else:
            if r < 0:
                    k = -k
                    hk = -hk
            if abs(r) < 1:
                aas = (1 - r) * (1 + r); a = numpy.sqrt(aas); bs = (h - k) ** 2;
                c = (4 - hk) / 8 ; d = (12 - hk) / 16; asr = -(bs / aas + hk) / 2;
                if asr > -100:
                    bvn = a * numpy.exp(asr) * (1 - c * (bs - aas) * (1 - d * bs / 5) / 3 + c * d * aas * aas / 5);
                if -hk < 100:
                    b = numpy.sqrt(bs); sp = numpy.sqrt(twopi) * stats.norm.cdf(-b / a);
                    bvn = bvn - numpy.exp(-hk / 2) * sp * b * (1 - c * bs * (1 - d * bs / 5) / 3);
                a = a / 2;
                for i in range(lg):
                    for iis in range(-1 , 3 , 2):
                        xs = (a * (iis * x[i] + 1)) ** 2; rs = numpy.sqrt(1 - xs);
                        asr = -(bs / xs + hk) / 2;
                        if asr > -100:
                            sp = (1 + c * xs * (1 + d * xs));
                            ep = numpy.exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs;
                            bvn = bvn + a * w[i] * numpy.exp(asr) * (ep - sp);
                bvn = -bvn / twopi;
            if r > 0: bvn = bvn + stats.norm.cdf(-max(h, k))
            if r < 0: bvn = -bvn + max(0, stats.norm.cdf(-h) - stats.norm.cdf(-k))
        p = max(0, min(1, bvn));
        return p

    def lower_MC(self, dh, dk, r, n=100000):
        """
            Computes bivariate stats.normal probabilities; lowerMC calculates the probability that x < dh and y < dk
            using a Monte Carlo approximation of n samples. The function only calls upperMC(-dh, -dk, r, n).

            \param dh 1st lower integration limit
            \param dk 2nd lower integration limit
            \param r   correlation coefficient
            \param n   sample size
        """
        return self.upper_MC(-dh, -dk, r, n)

    def upper_MC(self, dh, dk, r, n=100000):
        """
            Computes bivariate normal probabilities; upperMC calculates the probability that x > dh and y > dk. 
            This function is a simple MC evaluation used to cross-check the DW approximation algorithms.
        
            \param dh 1st lower integration limit
            \param dk 2nd lower integration limit
            \param r   correlation coefficient
            \param n   sample size
        
        """
        p = 0
        for i in range(n):
            v1 = numpy.random.normal(0, 1)
            v2 = r * v1 + numpy.sqrt(1 - r * r) * numpy.random.normal(0, 1)
            if v1 > dh and v2 > dk:p += 1
        return p / float(n)

bvnorm = _bvnorm(name='bvnorm', longname='A bivariate normal', shapes='r')
