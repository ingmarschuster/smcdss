#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with linear constraints. \namespace binary.product_cube """

import numpy

import binary.product as product
import binary.wrapper as wrapper

class CubeBinary(product.ProductBinary):
    """ Binary parametric family with linear constraints."""

    name = 'cube family'

    def __init__(self, p, size, name=name, long_name=__doc__):
        """ 
            Constructor.
            \param p mean vector
            \param name name
            \param long_name long_name
        """

        # call super constructor
        super(CubeBinary, self).__init__(p=p, name=name, long_name=long_name)

        # add module
        self.py_wrapper = wrapper.product_cube()
        self.pp_modules += ('binary.product_cube',)

        self.size

    @classmethod
    def _rvslpmf_all(cls, p, size, Y, U):
        """ 
            Log-probability mass function.
            \param Y binary vector
            \param param parameters
            \return log-probabilities
        """
        return None

    @classmethod
    def random(cls, d):
        """ 
            Construct a random family for testing.
            \param d dimension
        """
        return cls(p=0.01 + numpy.random.random(d) * 0.98, size=numpy.random.randint(1, d + 1))

    @classmethod
    def uniform(cls, d):
        """ 
            Construct a random product model for testing.
            \param d dimension
        """
        return cls(d=d, p=numpy.ones(d) * 0.5, size=numpy.random.randint(1, d + 1))

    def _getMean(self):
        """ Get expected value of instance. \return p-vector """
        return self.p * numpy.ones(self.d)

    def _getRandom(self, eps=0.0):
        return range(self.d)
