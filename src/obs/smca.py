#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Sequential Monte Carlo annealing.
"""

"""
@namespace obs.smca
$Author: christian.a.schafer@gmail.com $
$Rev: 144 $
$Date: 2011-05-12 19:12:23 +0200 (jeu., 12 mai 2011) $
@details
"""

import numpy
import time

import ubqo
import utils
import binary
from ibs.smc import ParticleSystem
from bf import solve_bf

class smca(ubqo.ubqo):
    name = 'SMCL'
    header = ['TYPE']
    def run(self):
        return solve_smca(f=binary.qu_exponential.QuExpBinary(self.A), v=self.v)

def solve_smca(f, v, verbose=True):

    t = time.time()
    v.update({'f':f, 'RUN_VERBOSE':False})
    print '\nRunning SMC-Annealing using %d particles using %s and %s...' % \
                (v['SMC_N_PARTICLES'], v['SMC_BINARY_MODEL'], v['SMC_CONDITIONING'])
    ps = ParticleSystem(v)
    model = ps.prop
    best_obj, best_soln = -numpy.inf, None
    bf = int(numpy.log(2 * ps.n) / numpy.log(2)) + 2

    # run sequential MC scheme
    while True:

        # progress ratio estimate
        #r = min((model.d - len(model.getRandom(0.05))) / float(model.d), 1.0)
        r = 1 - ps.pD

        # show progress bar
        current_obj, current_soln = ps.get_max()
        if best_obj < current_obj:
            best_soln = current_soln
            best_obj = current_obj
        if verbose: utils.auxi.progress(r, ' %03i, obj: %.1f, time: %s, rho: %.2f' % (len(model.r), best_obj, utils.auxi.time(time.time() - t), ps.rho))

        # check if dimension is sufficiently reduced
        if len(model.r) < bf:
            v = solve_bf(f=f, best_obj=best_obj, gamma=numpy.array([x > 0.5 for x in model.p]), index=model.r)
            best_obj, best_soln = v['obj'], v['soln']
            if verbose: utils.auxi.progress(1.0, ' %03i, obj: %.1f, time: %s, rho: %.2f' % (len(model.r), best_obj, utils.auxi.time(time.time() - t), ps.rho))
            break

        if ps.pD <= 5.0 / float(ps.n): break

        #ps.fit_proposal()
        ps.condition()
        ps.reweight_obs()
    if verbose:
        print '\nac', ps.r_ac, '\nns', ps.r_ns, '\npd', ps.r_pd

    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def main():
    pass

if __name__ == "__main__":
    main()
