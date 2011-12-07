#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Run a unit test sampling from all parametric families."""

"""
\namespace binary.unittest
$Author: christian.a.schafer@gmail.com $
$Rev: 152 $
$Date: 2011-10-10 10:51:51 +0200 (Mo, 10 Okt 2011) $
"""

from binary.product import ProductBinary, PositiveProductBinary, ConstrProductBinary, EquableProductBinary, LimitedProductBinary
from binary.logistic_cond import LogisticCondBinary

def main():

    for generator_class in [EquableProductBinary,
                            ProductBinary,
                            PositiveProductBinary,
                            ConstrProductBinary,
                            LimitedProductBinary,
                            LogisticCondBinary,
                            ]:

        generator = generator_class.random(d=8)
        print '\n' + 50 * '*' + '\n' + generator.name
        print generator
        print generator.rvstest(1000, start_jobserver=False)
        print generator.marginals(start_jobserver=False)

if __name__ == "__main__":
    main()
