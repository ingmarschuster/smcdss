#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family obtained via dichotomizing a multivariate Student.
    \namespace binary.copula_student
    \details The correlation structure of the model is limited by the constraints of the elliptic Student copula.
"""

import binary.copula as copula
import binary.wrapper as wrapper

import numpy
import scipy.stats as stats

class StudentCopulaBinary(copula.CopulaBinary):
    """ Binary parametric family obtained via dichotomizing a multivariate Student. """

    NU = 3.0

    def __init__(self, p, R, delta=None, name='Student copula family', long_name=__doc__):
        """ 
            Constructor.
            \param p mean
            \param R correlation matrix
        """

        # call super constructor
        super(StudentCopulaBinary, self).__init__(p, R, name=name, long_name=long_name)

        self.py_wrapper = wrapper.copula_student()

        # add modules
        self.pp_modules += ('binary.copula_student',)

    @classmethod
    def _rvs(cls, V, mu, C):
        """ 
            Generates a random variable.
            \param V normal variables
            \param param parameters
            \return binary variables
        """
        d = V.shape[1] - 1
        Y = numpy.empty((V.shape[0], d), dtype=bool)
        for k in xrange(V.shape[0]):
            Y[k] = mu > (numpy.dot(C, V[k, :d]) * numpy.sqrt(cls.NU / V[k, d]))
        return Y

    def _rvsbase(self, size):
        return numpy.hstack((numpy.random.normal(size=(size, self.d)),
                             numpy.random.chisquare(size=(size, 1), df=self.NU)))

    @classmethod
    def aux_cdf(cls, x, r):
        return bvt.cdf(x, r, cls.NU)

    @classmethod
    def aux_ppf(cls, p):
        return stats.t.ppf(p, cls.NU)


#-------------------------------------------------------------- bivariate student t

class _bvt(stats.rv_continuous):
    """
        Bivariate student t distribution with correlation coefficient r.
    """

    def rvs(self, r=0):
        """
            \param r correlation coefficient
            \param nu degrees of freedom
            \return random bivariate student t
        """
        g = numpy.random.normal(0, 1)
        x = numpy.array([g, r * g + numpy.sqrt(1 - r * r) * numpy.random.normal(0, 1)])
        return x * numpy.sqrt(self.NU / numpy.random.chisquare(self.NU))

    def cdf(self, x, r=0, nu=3.0):
        """
            Computes the bivariate Student t cumulative distribution function,
            i.e. the probability that X < x and Y < y. The function only calls lowerDS(x, y, r, nu).
            \param x value
            \param r correlation coefficient
            \param nu degrees of freedom
        """
        return self.lower_DS(nu, x[0], x[1], r=r)

    def lower_DS(self, nu=None, dh=None, dk=None, r=None, eps=1e-10):
        """
            A function for computing bivariate t probabilities.
            bvtl calculates the probability that x < dh and y < dk
           
            \param nu integer number of degrees of freedom, nu < 1, gives normal case
            \param dh 1st upper integration limit
            \param dk 2nd upper integration limit
            \param r   correlation coefficient

            This function is based on the method described by Dunnett, C.W. and M. Sobel, (1954),
            A bivariate generalization of Student's t-distribution with tables for certain special cases,
            Biometrika 41, pp. 153-169.
            
            The code was adapted for python from the matlab routine by Alan Genz.
        """
        if nu < 1:
            raise ValueError

        elif dh == -numpy.inf or dk == -numpy.inf:
            p = 0.0

        elif dh == numpy.inf:
            if dk == numpy.inf:
                p = 1.0
            else:
                p = stats.t.cdf(dk, nu)

        elif dk == numpy.inf:
            p = stats.t.cdf(dh, nu)

        elif 1 - r < eps:
            p = stats.t.cdf(min(dh, dk), nu)

        elif r + 1 < eps:
            p = 0.0
            if dh > -dk:
                p = stats.t.cdf(dh, nu) - stats.t.cdf(-dk, nu)
        else:
            tpi = 2 * numpy.pi
            ors = 1 - r * r
            hrk = dh - r * dk
            krh = dk - r * dh

            if abs(hrk) + ors > 0:
                xnhk = hrk ** 2 / (hrk ** 2 + ors * (nu + dk ** 2))
                xnkh = krh ** 2 / (krh ** 2 + ors * (nu + dh ** 2))
            else:
                xnhk = 0
                xnkh = 0

            hs = numpy.sign(dh - r * dk)
            ks = numpy.sign(dk - r * dh)

            if nu % 2 == 0:
                bvt = numpy.arctan2(numpy.sqrt(ors), -r) / tpi
                gmph = dh / numpy.sqrt(16 * (nu + dh ** 2))
                gmpk = dk / numpy.sqrt(16 * (nu + dk ** 2))

                btnckh = 2 * numpy.arctan2(numpy.sqrt(xnkh), numpy.sqrt(1 - xnkh)) / numpy.pi
                btpdkh = 2 * numpy.sqrt(xnkh * (1 - xnkh)) / numpy.pi
                btnchk = 2 * numpy.arctan2(numpy.sqrt(xnhk), numpy.sqrt(1 - xnhk)) / numpy.pi
                btpdhk = 2 * numpy.sqrt(xnhk * (1 - xnhk)) / numpy.pi

                for j in xrange(1, int(nu) / 2 + 1):
                    bvt = bvt + gmph * (1 + ks * btnckh)
                    bvt = bvt + gmpk * (1 + hs * btnchk)
                    btnckh = btnckh + btpdkh
                    btpdkh = 2 * j * btpdkh * (1 - xnkh) / float(2 * j + 1)
                    btnchk = btnchk + btpdhk
                    btpdhk = 2 * j * btpdhk * (1 - xnhk) / float(2 * j + 1)

                    gmph = gmph * (j - 1 / 2.0) / float(j * (1 + dh ** 2 / nu))
                    gmpk = gmpk * (j - 1 / 2.0) / float(j * (1 + dk ** 2 / nu))

            else:
                qhrk = numpy.sqrt(dh ** 2 + dk ** 2 - 2 * r * dh * dk + nu * ors)

                hkrn = dh * dk + r * nu
                hkn = dh * dk - nu
                hpk = dh + dk

                bvt = numpy.arctan2(-numpy.sqrt(nu) * (hkn * qhrk + hpk * hkrn), hkn * hkrn - nu * hpk * qhrk) / tpi
                if bvt < -10 * eps:
                    bvt = bvt + 1

                gmph = dh / (tpi * numpy.sqrt(nu) * (1 + dh ** 2 / nu))
                gmpk = dk / (tpi * numpy.sqrt(nu) * (1 + dk ** 2 / nu))
                btnckh = numpy.sqrt(xnkh)
                btpdkh = btnckh

                btnchk = numpy.sqrt(xnhk)
                btpdhk = btnchk

                for j in xrange(1, (int(nu) - 1) / 2 + 1):

                    bvt = bvt + gmph * (1 + ks * btnckh)
                    bvt = bvt + gmpk * (1 + hs * btnchk)
                    btpdkh = (2 * j - 1) * btpdkh * (1 - xnkh) / float(2 * j)
                    btnckh = btnckh + btpdkh
                    btpdhk = (2 * j - 1) * btpdhk * (1 - xnhk) / float(2 * j)
                    btnchk = btnchk + btpdhk
                    gmph = gmph * j / float((j + 1 / 2) * (1 + dh ** 2 / nu))
                    gmpk = gmpk * j / float((j + 1 / 2) * (1 + dk ** 2 / nu))
            p = bvt
        return p

    def lower_MC(self, nu, dh, dk, r, n=1e5):
        """
            Computes bivariate stats.normal probabilities; lowerMC calculates the probability that x < dh and y < dk
            using a Monte Carlo approximation of n samples. The function only calls upperMC(-dh, -dk, r, n).

            \param dh 1st lower integration limit
            \param dk 2nd lower integration limit
            \param r   correlation coefficient
            \param n   sample size
        """
        return self.upper_MC(nu, -dh, -dk, r, n)

    def upper_MC(self, nu, dh, dk, r, n=1e5):
        """
            Computes bivariate normal probabilities; upperMC calculates the probability that x > dh and y > dk. 
            This function is a simple MC evaluation used to cross-check the DW approximation algorithms.
        
            \param dh 1st lower integration limit
            \param dk 2nd lower integration limit
            \param r   correlation coefficient
            \param n   sample size
        
        """
        p = 0
        sr = numpy.sqrt(1 - r * r)
        for i in xrange(int(n)):
            v = numpy.random.normal(size=2)
            v[1] = r * v[0] + sr * numpy.random.normal()
            v *= numpy.sqrt(nu / numpy.random.chisquare(df=nu))
            if v[0] > dh and v[1] > dk:
                p += 1
        return p / float(n)

bvt = _bvt(name='bvt', longname='A bivariate t-distribution', shapes='r')