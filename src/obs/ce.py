#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Cross-entropy optimization.
"""

"""
@namespace obs.ce
$Author$
$Rev$
$Date$
@details
"""

from obs import *

class ce(ubqo.ubqo):
    header = []
    name = 'CE'
    def run(self):
        return solve_ce(f=binary.QuExpBinary(self.A),
                        n=int(self.v['CE_N_PARTICLES']),
                        model=self.v['CE_BINARY_MODEL'],
                        lag=self.v['CE_LAG'],
                        elite=self.v['CE_ELITE'],
                        job_server=self.v['JOB_SERVER'])

def solve_ce(f, n=5e4, model=binary.LogisticBinary, lag=0.2, elite=0.2, job_server=None, verbose=True):
    """ Finds a maximum via cross-entropy optimization.
        @param f function
        @param n number of particles
        @param model binary model
        @param lag parameter update lag
        @param elite elite fraction
        @param verbose verbose
    """

    t = time.time()
    bf = int(numpy.log(2 * n) / numpy.log(2)) + 1
    model = model.uniform(f.d)
    print 'running ce using ' + model.name

    d = utils.data.data()
    best_obj = -numpy.inf
    best_soln = numpy.zeros(f.d)

    # run optimization scheme
    for step in xrange(1, 100):

        d.sample(model, n, job_server)
        best_obj, best_soln = d.dichotomize_weights(f=f, fraction=elite)

        model.renew_from_data(sample=d, lag=lag, verbose=False)

        # progress ratio estimate
        r = (model.d - len(model.getRandom(0.05 + 0.5 / (step + 1)))) / float(model.d)

        # show progress bar
        if verbose: utils.format.progress(r, ' %02i, %03i, objective: %.1f, time %s' % (step, len(model.r), best_obj, utils.format.time(time.time() - t)))

        # check if dimension is sufficiently reduced
        if len(model.r) < bf:
            v = solve_bf(f=f, best_obj=best_obj, gamma=best_soln, index=model.r)
            best_obj, best_soln = v['obj'], v['soln']
            if verbose: utils.format.progress(1.0, ' %02i, %03i, objective: %.1f, time %s' % (step + 1, len(model.r), best_obj, utils.format.time(time.time() - t)))
            break
        d.clear(fraction=elite)

    if verbose: sys.stdout.write('\n')
    return {'obj' : best_obj, 'soln' : best_soln, 'time' : time.time() - t}

def main():
    pass

if __name__ == "__main__":
    main()
