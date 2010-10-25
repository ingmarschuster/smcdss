'''
Created on 28 oct. 2009

@author: cschafer
'''

from scipy.stats import norm, rv_continuous
from numpy import sqrt, random, exp, log, zeros, cos, sin, arcsin, arccos, arctan, pi

class bvnorm_gen(rv_continuous):
    
    def cdf(self, x, r=0):
        '''
        Computes the bivariate normal cumulative distribution function, i.e. the probability that X < x and Y < y. 
        
        The function only calls lowerDW(x, y, r).
        '''
        return self.lowerDW(x[0], x[1], r)
    
    def pdf(self, x, r=0):
        '''
        Computes the bivariate normal probability distribution function, i.e. the density at (x, y) 
        '''
        z = x[0] * x[0] - 2 * r * x[0] * x[1] + x[1] * x[1]
        return exp(-z / (2 * (1 - r * r))) / (2 * pi * sqrt(1 - r * r))
    
    def _rvs(self, r=0):
        v = random.normal(0, 1)
        return r * v + sqrt(1 - r * r) * random.normal(0, 1)
    
    def lowerDW(self, dh, dk, r):
        '''
        Computes bivariate normal probabilities; lowerDW calculates the probability that x < dh and y < dk using the Drezner-Wesolowsky approximation.
        
        The function only calls upperDW(-dh, -dk, r).
        '''
        return self.upperDW(-dh, -dk, r)
    
    def upperDW(self, dh, dk, r):
        '''
        Computes bivariate normal probabilities; upperDW calculates the probability that x > dh and y > dk. 
        
            dh 1st lower integration limit
            dk 2nd lower integration limit
            r   correlation coefficient
        
        This function is based on the method described by Z. Drezner and G.O. Wesolowsky, (1989), "On the computation of the bivariate normal integral", Journal of Statist. Comput. Simul. 35, pp. 101-107, with major modifications for double precision, for |r| close to 1.
        
        The code was adapted for python from the matlab routine by Alan Genz.
        '''
        twopi = 2 * pi
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
            hs = (h * h + k * k) / 2; asr = arcsin(r);
            for i in range(lg):
                sn = sin(asr * (1 - x[i]) / 2);
                bvn = bvn + w[i] * exp((sn * hk - hs) / (1 - sn * sn));
                sn = sin(asr * (1 + x[i]) / 2);
                bvn = bvn + w[i] * exp((sn * hk - hs) / (1 - sn * sn));
            bvn = bvn * asr / (4 * pi) + norm.cdf(-h) * norm.cdf(-k)
        else:
            if r < 0:
                    k = -k
                    hk = -hk
            if abs(r) < 1:
                aas = (1 - r) * (1 + r); a = sqrt(aas); bs = (h - k) ** 2;
                c = (4 - hk) / 8 ; d = (12 - hk) / 16; asr = -(bs / aas + hk) / 2;
                if asr > -100: 
                    bvn = a * exp(asr) * (1 - c * (bs - aas) * (1 - d * bs / 5) / 3 + c * d * aas * aas / 5);
                if - hk < 100:
                    b = sqrt(bs); sp = sqrt(twopi) * norm.cdf(-b / a);
                    bvn = bvn - exp(-hk / 2) * sp * b * (1 - c * bs * (1 - d * bs / 5) / 3);
                a = a / 2;
                for i in range(lg):
                    for iis in range(-1 , 3 , 2):
                        xs = (a * (iis * x[i] + 1)) ** 2; rs = sqrt(1 - xs);
                        asr = -(bs / xs + hk) / 2;
                        if asr > -100:
                            sp = (1 + c * xs * (1 + d * xs));
                            ep = exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs;
                            bvn = bvn + a * w[i] * exp(asr) * (ep - sp);
                bvn = -bvn / twopi;
            if r > 0: bvn = bvn + norm.cdf(-max(h, k))
            if r < 0: bvn = -bvn + max(0, norm.cdf(-h) - norm.cdf(-k))
        p = max(0, min(1, bvn));
        return p
    
    def lowerDI(dh, dk, r):
        '''
        Computes bivariate normal probabilities; lowerDW calculates the probability that x < dh and y < dk using the Drezner-Wesolowsky approximation.
        
        The function only calls upperDI(-dh, -dk, r).
        '''
        return upperDI(-dh, -dk, r)
    
    def upperDI(dh, dk, r):
        '''
        Computes bivariate normal probabilities; upperDW calculates the probability that x > dh and y > dk. 
        
            dh 1st lower integration limit
            dk 2nd lower integration limit
            r   correlation coefficient
        
        This function is based on the method described by D.R. Divgi, (1979), "Calculation of univariate and bivariate normal probability functions", Teh Annals of Statistics, Vol.7, No.4, pp. 903-910.
        
        The function does not yet deliver the correct values for every correlation coefficient r.
        '''
        R = sqrt((dh * dh + dk * dk - 2 * dh * dk * r) / (1 - r * r))
        if dh < 0 and dk < 0: c = 1
        else: c = 0
        if dh == 0 and dk == 0:
            t1 = arctan(sqrt((1 + r) / (1 - r)))
            t2 = t1
        else:
            t1 = arcsin(dh / R)
            t2 = arcsin(dk / R)
        if t1 * t2 < 0: sign = -1
        else: sign = 1
        p = sign * __W(R, sign * t1) + sign * __W(R, sign * t2) + c
        return p
    
    def __W(R, psi):    
        '''
        Auxiliary function for Divgi method
        '''
        if psi < 0: sign1 = -1; psi = pi + psi
        else: sign1 = 1
        if psi > pi / 2: Q = norm.cdf(-R * sin(psi)); sign2 = -1; psi = pi - psi
        else: Q = 0; sign2 = 1
        
        d = [1.253298042, -0.9997316607, 0.6250192459, -0.3281915667, 0.1470331965, \
           - 0.05494856177, 0.01629827794, -0.003591257830, 0.0005406619903, -0.0000489254061, 0.000001984741031]
    
        c = cos(psi)
        s = sin(psi)
        powers = zeros(12)
        powers[0] = psi
        powers[1] = s
        for i in range(2, 12):
            powers[i] = ((c ** (i - 1)) * s + (i - 1) * powers[i - 2]) / float(i)
        
        sum = 0
        for i in range(11):
            sum += d[i] * (R ** (i + 1)) * powers[i + 1]
        W = (1 / (2 * pi)) * exp(-R * R / 2) * (psi - sum)
        return sign1 * (Q + sign2 * W)      
    
    def lowerMC(self, dh, dk, r, n=100000):
        '''
        Computes bivariate normal probabilities; lowerMC calculates the probability that x < dh and y < dk using a Monte Carlo approximation of n samples.
        
        The function only calls upperMC(-dh, -dk, r, n).
        '''
        return self.upperMC(-dh, -dk, r, n)
    
    def upperMC(self, dh, dk, r, n=100000):
        '''
        Computes bivariate normal probabilities; upperMC calculates the probability that x > dh and y > dk. 
        
            dh 1st lower integration limit
            dk 2nd lower integration limit
            r   correlation coefficient
            n sample size (=100000)
        
        This function is a simple MC evaluation used to cross-check the approximation algorithms DW and DI.
        '''
        p = 0
        for i in range(n):
            v1 = random.normal(0, 1)
            v2 = r * v1 + sqrt(1 - r * r) * random.normal(0, 1)
            if v1 > dh and v2 > dk:p += 1
        return p / float(n)
    
bvnorm = bvnorm_gen(name='bvnorm', longname='A bivariate normal', shapes='r', extradoc="""

Bivariate normal distribution with correlation r.

normal.pdf(x,y) = exp(-(x*x-2*r*x*y+y*y)/(2*(1-r*r))) / (2*pi*sqrt(1-r*r))
""")
