#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Wrapper classes for use with parallel python. \namespace binary.wrapper"""

import numpy
import binary

class wrapper():
    def rvslpmf(self, U, param):
        Y = self.rvs(U=U, param=param)
        return Y, self.lpmf(Y, param=param)

class product(wrapper):
    def lpmf(self, Y, param):
        return binary.product.ProductBinary._lpmf(numpy.array(Y, dtype=numpy.int8), p=param.p)
    def rvs(self, U, param):
        return binary.product.ProductBinary._rvs(U=U, p=param.p)

class product_exchangeable(wrapper):
    def lpmf(self, Y, param):
        return binary.product_exchangeable.ExchangeableBinary._lpmf(Y=Y, logit_p=param.logit_p)
    def rvs(self, U, param):
        return binary.product_exchangeable.ExchangeableBinary._rvs(U=U, p=param.p)

class product_positive(wrapper):
    def lpmf(self, Y, param):
        return binary.product_positive.PositiveBinary._rvslpmf_all(
                    Y=Y, log_p=param.log_p, log_q=param.log_q, log_c=param.log_c)[1]
    def rvs(self, U, param):
        return self.rvslpmf(U=U, param=param)[0]
    def rvslpmf(self, U, param):
        return binary.product_positive.PositiveBinary._rvslpmf_all(
                    U=U, log_p=param.log_p, log_q=param.log_q, log_c=param.log_c)

class product_constrained(wrapper):
    def lpmf(self, Y, param):
        return binary.product_constrained.ConstrProductBinary._lpmf(
                    Y=Y, p=param.p, constrained=param.constrained)
    def rvs(self, U, param):
        return binary.product_constrained.ConstrProductBinary._rvs(
                    U=U, p=param.p, constrained=param.constrained, free=param.free)

class product_limited(wrapper):
    def lpmf(self, Y, param):
        return binary.product_limited.LimitedBinary._lpmf(Y, param.q, param.logit_p)
    def rvs(self, U, param):
        return binary.product_limited.LimitedBinary._rvs(U, param.b)

class product_cube(wrapper):
    def lpmf(self, Y, param):
        return binary.product_cube.CubeBinary._rvslpmf_all(
                p=param.p, size=param.size, Y=Y)[1]
    def rvs(self, U, param):
        return self.rvslpmf(U, param)[0]
    def rvslpmf(self, U, param):
        return binary.product_cube.CubeBinary._rvslpmf_all(
                p=param.p, size=param.size, U=U)

class conditionals_logistic(wrapper):
    def lpmf(self, Y, param):
        return binary.conditionals_logistic.LogisticCondBinary._rvslpmf_all(
                A=param.A, U=None, Y=numpy.array(Y, dtype=numpy.int8))[1]
    def rvs(self, U, param):
        return self.rvslpmf(U, param)[0]
    def rvslpmf(self, U, param):
        return binary.conditionals_logistic.LogisticCondBinary._rvslpmf_all(
                A=param.A, U=U)

class conditionals_linear(wrapper):
    def lpmf(self, Y, param):
        return binary.conditionals_linear.LinearCondBinary._rvslpmf_all(
                A=param.A, U=None, Y=numpy.array(Y, dtype=numpy.int8))[1]
    def rvs(self, U, param):
        return self.rvslpmf(U, param)[0]
    def rvslpmf(self, U, param):
        return binary.conditionals_linear.LinearCondBinary._rvslpmf_all(
                A=param.A, U=U)

class conditionals_arctan(wrapper):
    def lpmf(self, Y, param):
        return binary.conditionals_arctan.ArctanCondBinary._rvslpmf_all(
                A=param.A, Y=numpy.array(Y, dtype=numpy.int8))[1]
    def rvs(self, U, param):
        return self.rvslpmf(U, param)[0]
    def rvslpmf(self, U, param):
        return binary.conditionals_arctan.ArctanCondBinary._rvslpmf_all(
                A=param.A, U=U)

class quadratic_linear(wrapper):
    def lpmf(self, Y, param):
        return binary.quadratic_linear.QuLinearBinary._rvslpmf_all(
                A=param.A, p=param.p, Y=numpy.array(Y, dtype=numpy.int8))[1]
    def rvs(self, U, param):
        return self.rvslpmf(U, param)[0]
    def rvslpmf(self, U, param):
        return binary.quadratic_linear.QuLinearBinary._rvslpmf_all(
                A=param.A, p=param.p, U=U)

class quadratic_exponential(wrapper):
    def lpmf(self, Y, param):
        return binary.quadratic_exponential.QuExpBinary._lpmf(Y, A=param.A)

class copula_gaussian(wrapper):
    def lpmf(self, Y, param):
        return None
    def rvs(self, V, param):
        return binary.copula_gaussian.GaussianCopulaBinary._rvs(V, param.mu, param.C)
    def rvslpmf(self, V, param):
        return self.rvs(V, param), None

class copula_student(wrapper):
    def lpmf(self, Y, param):
        return None
    def rvs(self, V, param):
        return  binary.copula_student.StudentCopulaBinary._rvs(V, param.mu, param.C)
    def rvslpmf(self, V, param):
        return self.rvs(V, param), None

class posterior_bic(wrapper):
    def lpmf(self, Y, param):
        return binary.posterior_bic.PosteriorBIC._lpmf(Y, param)

class posterior_bvs(wrapper):
    def lpmf(self, Y, param):
        return binary.posterior_bvs.PosteriorBVS._lpmf(Y, param)