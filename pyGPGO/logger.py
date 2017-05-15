class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class EventLogger:
    def __init__(self, gpgo):
        self.gpgo = gpgo
        self.header = 'Evaluation \t Proposed point \t  Current eval. \t Best eval.'
        self.template = '{:6} \t {}. \t  {:6} \t {:6}'
        print(self.header)

    def _printCurrent(self, gpgo):
        eval = str(len(gpgo.GP.y) - gpgo.init_evals)
        proposed = str(gpgo.best)
        curr_eval = str(gpgo.GP.y[-1])
        curr_best = str(gpgo.tau)
        if float(curr_eval) >= float(curr_best):
            curr_eval = bcolors.OKGREEN + curr_eval + bcolors.ENDC
        print(self.template.format(eval, proposed, curr_eval, curr_best))

    def _printInit(self, gpgo):
        for init_eval in range(gpgo.init_evals):
            print(self.template.format('init', gpgo.GP.X[init_eval], gpgo.GP.y[init_eval], gpgo.tau))

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from pyGPGO.covfunc import squaredExponential
    from pyGPGO.surrogates.GaussianProcess import GaussianProcess
    from pyGPGO.acquisition import Acquisition
    from pyGPGO.GPGO import GPGO

    np.random.seed(20)

    def f(x):
        return -((6*x-2)**2*np.sin(12*x-4))

    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode = 'ExpectedImprovement')

    params = {'x': ('cont', (0, 1))}
    gpgo = GPGO(gp, acq, f, params)
    gpgo.run(max_iter = 10)
    print(gpgo.getResult())
