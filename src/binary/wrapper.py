#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with independent components."""

"""
\namespace binary.wrapper
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

import numpy
import binary

class wrapper():
    def rvslpmf(self, U, param):
        Y = self.rvs(U=U, param=param)
        return Y, self.lpmf(Y, param=param)

class equable_product(wrapper):
    """ Wrapper class for equable product family."""

    def lpmf(self, Y, param):
        return binary.product.EquableProductBinary._lpmf(Y=Y, logit_p=param.logit_p)
    def rvs(self, U, param):
        return binary.product.EquableProductBinary._rvs(U=U, p=param.p)

class product(wrapper):
    """ Wrapper class for product family."""

    def lpmf(self, Y, param):
        return binary.product.ProductBinary._lpmf(numpy.array(Y, dtype=numpy.int8), p=param.p)
    def rvs(self, U, param):
        return binary.product.ProductBinary._rvs(U=U, p=param.p)

class positive_product(wrapper):
    """ Wrapper class for positive product family."""

    def lpmf(self, Y, param):
        return binary.product.PositiveProductBinary._rvslpmf_all(
                    Y=Y, log_p=param.log_p, log_q=param.log_q, log_c=param.log_c)[1]
    def rvs(self, U, param):
        return self.rvslpmf(U=U, param=param)[0]
    def rvslpmf(self, U, param):
        return binary.product.PositiveProductBinary._rvslpmf_all(
                    U=U, log_p=param.log_p, log_q=param.log_q, log_c=param.log_c)

class constrained_product(wrapper):
    """ Wrapper class for logistic conditionals family."""

    def lpmf(self, Y, param):
        return binary.product.ConstrProductBinary._lpmf(
                    Y, param.p, param.constrained)
    def rvs(self, U, param):
        return binary.product.ConstrProductBinary._rvs(
                    U, param.p, param.constrained, param.free)


class limited_product(wrapper):
    """ Wrapper class for logistic conditionals family."""

    def lpmf(self, Y, param):
        return binary.product.LimitedProductBinary._lpmf(Y, param.q, param.logit_p)
    def rvs(self, U, param):
        return binary.product.LimitedProductBinary._rvs(U, param.b)


class logistic_cond(wrapper):
    """ Wrapper class for logistic conditionals family."""

    def lpmf(self, Y, param):
        return binary.logistic_cond.LogisticCondBinary._rvslpmf_all(
                Beta=param.Beta, U=None, Y=numpy.array(Y, dtype=numpy.int8))[1]
    def rvs(self, U, param):
        return self.rvslpmf(U, param)[0]
    def rvslpmf(self, U, param):
        return binary.logistic_cond.LogisticCondBinary._rvslpmf_all(
                Beta=param.Beta, U=U)
    def calc_log_regr(self, y, X, XW, init, w=None, verbose=False):
        return binary.logistic_cond.calc_log_regr(y, X, XW, init, w, verbose)

class linear_cond(wrapper):
    """ Wrapper class for linear conditionals family."""
    def lpmf(self, Y, param):
        return binary.linear_cond.LinearCondBinary._rvslpmf_all(
                param.Beta, Y=numpy.array(Y, dtype=numpy.int8))[1]
    def rvs(self, U, param):
        return self.rvslpmf(U, param)[0]
    def rvslpmf(self, U, param):
        return binary.linear_cond.LinearCondBinary._rvslpmf_all(
                param.Beta, U=U)


class qu_linear(wrapper):
    """ Wrapper class for logistic conditionals family."""

    def lpmf(self, Y, param):
        return binary.qu_linear.QuLinearBinary._rvslpmf_all(
                param.Beta, param.p, Y=Y)[1]
    def rvs(self, U, param):
        return self.rvslpmf(U, param)[0]
    def rvslpmf(self, U, param):
        return binary.qu_linear.QuLinearBinary._rvslpmf_all(
                param.Beta, param.p, U=U)


class qu_exponential(wrapper):
    """ Wrapper class for quadratic exponential family."""

    def lpmf(self, Y, param):
        return binary.qu_exponential.QuExpBinary._lpmf(Y, A=param.A)


class gaussian_copula(wrapper):
    """ Wrapper class for logistic conditionals family."""

    def lpmf(self, Y, param):
        return None
    def rvs(self, V, param):
        return binary.gaussian_copula.GaussianCopulaBinary._rvs(V, param.mu, param.C)
    def rvslpmf(self, V, param):
        return self.rvs(V, param), None

class student_copula(wrapper):
    """ Wrapper class for logistic conditionals family."""

    def lpmf(self, Y, param):
        return None
    def rvs(self, V, param):
        return binary.student_copula.StudentCopulaBinary._rvs(V, param.mu, param.C)
    def rvslpmf(self, V, param):
        return self.rvs(V, param), None


class posterior(wrapper):
    """ Wrapper class for logistic conditionals family."""

    def lpmf(self, Y, param):
        return binary.posterior.Posterior._lpmf(Y, param)
    def rvs(self, V, param):
        raise NotImplementedError('Random variable generation not possible.')
    def rvslpmf(self, V, param):
        raise NotImplementedError('Random variable generation not possible.')
