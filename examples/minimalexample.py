#######################################
# pyGPGO examples
# minimalexample: A minimal working pyGPGO example.
#######################################

import numpy as np
import matplotlib.pyplot as plt
from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.GPGO import GPGO


def drawFun(f):
    x = np.linspace(0, 1, 1000)
    plt.plot(x, f(x))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    np.random.seed(20)
    def f(x):
        return -((6*x-2)**2*np.sin(12*x-4))

    drawFun(f)

    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode = 'ExpectedImprovement')

    params = {'x': ('cont', (0, 1))}
    gpgo = GPGO(gp, acq, f, params)
    gpgo.run(max_iter = 10)
    print(gpgo.getResult())
