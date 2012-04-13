#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Binary parametric family with arctan conditionals. \namespace binary.conditionals_arctan"""

import numpy

import binary.conditionals as conditionals
import binary.wrapper as wrapper

class ArctanCondBinary(conditionals.ConditionalsBinary):
    """ Binary parametric family with arctan conditionals. """

    def __init__(self, A, name='arctan conditionals family', long_name=__doc__):
        """ 
            Constructor.
            \param A Lower triangular matrix holding regression coefficients
            \param name name
            \param long_name long name
        """

        # call super constructor
        super(ArctanCondBinary, self).__init__(A=A, name=name, long_name=long_name)

        # add modules
        self.pp_modules = ('numpy', 'scipy.linalg', 'binary.conditionals_arctan',)

        self.py_wrapper = wrapper.conditionals_arctan()

    @classmethod
    def link(cls, x):
        """ Arctan function \return cdf of Cauchy """
        return 0.5 + (numpy.arctan(x) / numpy.pi)

    @classmethod
    def dlink(cls, x):
        """ Derivative of arctan function \return pdf of Cauchy """
        return 1.0 / ((1.0 + x * x) * numpy.pi)

    @classmethod
    def ilink(cls, p):
        """ Inverse of arctan function \return ppf of Cauchy """
        return numpy.tan(numpy.pi * (p - 0.5))
