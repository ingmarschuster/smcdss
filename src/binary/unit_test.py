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
from binary.constrained import ConstrSizeBinary, ConstrInteractionBinary

def main():

    for generator_class in [#ProductBinary,
                            #PosProductBinary,
                            #LogisticCondBinary,
                            ConstrInteractionBinary]:

        generator = generator_class.random(d=15, p=0.75)
        print '\n' + 50 * '*' + '\n' + generator.name
        print generator
        print generator.rvstest(50000, start_jobserver=False)
        print generator.marginals(start_jobserver=False)

if __name__ == "__main__":
    main()
