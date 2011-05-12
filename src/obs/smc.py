#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Sequential Monte Carlo optimization.
"""

"""
@namespace obs.smc
$Author$
$Rev$
$Date$
@details
"""

from ibs.smc import ParticleSystem
from obs import *

class smc(ubqo.ubqo):
    name = 'SMC'
    header = ['TYPE']
    def run(self):
        return solve_smc(f=binary.QuExpBinary(self.A), v=self.v)

def solve_smc(f, v, verbose=True):

    t = time.time()
    v.update({'f':f, 'RUN_VERBOSE':False})
    print '\nrunning smc using %s and %s' % (v['SMC_BINARY_MODEL'].__name__, v['SMC_CONDITIONING'])
    ps = ParticleSystem(v)
    model = ps.prop
    best_obj, best_soln = -numpy.inf, None
    bf = int(numpy.log(2 * ps.n) / numpy.log(2)) + 2

    # run sequential MC scheme
    while True:

        # progress ratio estimate
        r = max(ps.rho, (model.d - len(model.getRandom(0.05))) / float(model.d))

        # show progress bar
        current_obj, current_soln = ps.getMax()
        if best_obj < current_obj:
            best_soln = current_soln
            best_obj = current_obj
        if verbose: utils.format.progress(r, ' %03i, objective: %.1f, time %s' % (len(model.r), best_obj, utils.format.time(time.time() - t)))

        # check if dimension is sufficiently reduced
        if len(model.r) < bf:
            v = solve_bf(f=f, best_obj=best_obj, gamma=numpy.array([x > 0.5 for x in model.p]), index=model.r)
            best_obj, best_soln = v['obj'], v['soln']
            if verbose: utils.format.progress(1.0, ' %03i, objective: %.1f, time %s' % (len(model.r), best_obj, utils.format.time(time.time() - t)))
            break

        ps.fit_proposal()
        ps.condition()
        ps.reweight()

    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def main():
    pass

if __name__ == "__main__":
    main()
