#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Run a unit test sampling from all parametric families."""

"""
\namespace binary.unittest
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

from binary.pos_product import PosProductBinary
from binary.product import ProductBinary
from binary.logistic_cond import LogisticCondBinary
from binary.uniform import UniformBinary

def main():

    for generator_class in [ProductBinary,
                            PosProductBinary,
                            LogisticCondBinary,
                            UniformBinary]:

        generator = generator_class.random(5)
        print '\n' + 50 * '*' + '\n' + generator.name
        print generator
        print generator.rvstest(500)
        print generator.marginals()

if __name__ == "__main__":
    main()
