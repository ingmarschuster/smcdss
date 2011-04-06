#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#    $Author: Christian Schäfer
#    $Date: 2011-04-04 15:26:12 +0200 (lun., 04 avr. 2011) $

__version__ = "$Revision: 100 $"

from ibs.smc import ParticleSystem
from obs import *

class smc(ubqo.ubqo):
    name = 'SMC'
    header = []
    def run(self):
        return solve_smc(f=binary.QuExpBinary(self.A), v=self.v)

def solve_smc(f, v, verbose=True):

    t = time.time()
    v.update({'f':f, 'RUN_VERBOSE':False})
    ps = ParticleSystem(v)
    model = ps.prop
    print 'running smc using ' + model.name
    bf = int(numpy.log(2 * ps.n) / numpy.log(2)) + 1

    # run sequential MC scheme
    while True:

        # progress ratio estimate
        r = max(ps.rho, (model.d - len(model.getRandom(0.05))) / float(model.d))

        # show progress bar
        best_soln = numpy.array([x > 0.5 for x in model.p])
        best_obj = ps.f.lpmf(best_soln)
        if verbose: utils.format.progress(r, ' %03i, objective: %.1f' % (len(model.r), best_obj))

        # check if dimension is sufficiently reduced
        if len(model.r) < bf:
            v = solve_bf(f=f, best_obj=best_obj, gamma=best_soln, index=model.r)
            best_obj, best_soln = v['obj'], v['soln']
            if verbose: utils.format.progress(1.0, ' %03i, objective: %.1f' % (len(model.r), best_obj))
            break

        ps.fit_proposal()
        ps.resample()
        ps.move()
        ps.reweight()

    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def main():
    pass

if __name__ == "__main__":
    main()
