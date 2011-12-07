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

class product():
    """ Wrapper class for product family."""

    def lpmf(self, gamma, param):
        return binary.product.ProductBinary._lpmf(numpy.array(gamma, dtype=numpy.int8), param['p'])
    def rvs(self, U, param):
        return binary.product.ProductBinary._rvs(U, param)
    def rvslpmf(self, U, param):
        Y = binary.product.ProductBinary._rvs(U, param)
        return Y, binary.product.ProductBinary._lpmf(numpy.array(Y, dtype=numpy.int8), param['p'])


class positive_product():
    """ Wrapper class for positive product family."""

    def lpmf(self, gamma, param):
        return binary.product.PositiveProductBinary._posproduct_all(param, gamma=gamma)[1]
    def rvs(self, U, param):
        return binary.product.PositiveProductBinary._posproduct_all(param, U=U)[0]
    def rvslpmf(self, U, param):
        return binary.product.PositiveProductBinary._posproduct_all(param, U=U)


class logistic_cond():
    """ Wrapper class for logistic conditionals family."""

    def lpmf(self, gamma, param):
        return binary.logistic_cond.LogisticCondBinary._logistic_cond_all(
                Beta=param['Beta'], U=None, gamma=numpy.array(gamma, dtype=numpy.int8))[1]
    def rvs(self, U, param):
        return binary.logistic_cond.LogisticCondBinary._logistic_cond_all(
                Beta=param['Beta'], U=U, gamma=None)[0]
    def rvslpmf(self, U, param):
        return binary.logistic_cond.LogisticCondBinary._logistic_cond_all(
                Beta=param['Beta'], U=U, gamma=None)

class equable_product():
    """ Wrapper class for equable product family."""

    def lpmf(self, gamma, param):
        return binary.product.EquableProductBinary._lpmf(gamma, param)
    def rvs(self, U, param):
        return binary.product.EquableProductBinary._rvs(U, param)
    def rvslpmf(self, U, param):
        return binary.product.EquableProductBinary._rvslpmf(U, param)

class constrained_product():
    """ Wrapper class for logistic conditionals family."""

    def lpmf(self, gamma, param):
        return binary.product.ConstrProductBinary._lpmf(gamma, param)
    def rvs(self, U, param):
        return binary.product.ConstrProductBinary._rvs(U, param)
    def rvslpmf(self, U, param):
        return binary.product.ConstrProductBinary._rvslpmf(U, param)
    
class limited_product():
    """ Wrapper class for logistic conditionals family."""

    def lpmf(self, gamma, param):
        return binary.product.LimitedProductBinary._lpmf(gamma, param)
    def rvs(self, U, param):
        return binary.product.LimitedProductBinary._rvs(U, param)
    def rvslpmf(self, U, param):
        return binary.product.LimitedProductBinary._rvslpmf(U, param)
