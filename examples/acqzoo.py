#######################################
# pyGPGO examples
# acqzoo: shows the behaviour of different
# acquisition functions on a GP surrogate
# for a sine-like function
#######################################


import numpy as np
import matplotlib.pyplot as plt
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.GPGO import GPGO


def plotGPGO(gpgo, param, index, new=True):
    param_value = list(param.values())[0][1]
    x_test = np.linspace(param_value[0], param_value[1], 1000).reshape((1000, 1))
    y_hat, y_var = gpgo.GP.predict(x_test, return_std=True)
    std = np.sqrt(y_var)
    l, u = y_hat - 1.96 * std, y_hat + 1.96 * std
    if new:
        plt.figure()
        plt.subplot(5, 1, 1)
        plt.fill_between(x_test.flatten(), l, u, alpha=0.2)
        plt.plot(x_test.flatten(), y_hat)
    plt.subplot(5, 1, index)
    a = np.array([-gpgo._acqWrapper(np.atleast_1d(x)) for x in x_test]).flatten()
    plt.plot(x_test, a, color=colors[index - 2], label=acq_titles[index - 2])
    gpgo._optimizeAcq(method='L-BFGS-B', n_start=1000)
    plt.axvline(x=gpgo.best)
    plt.legend(loc=0)


if __name__ == '__main__':
    def f(x):
        return (np.sin(x))

    acq_1 = Acquisition(mode='ExpectedImprovement')
    acq_2 = Acquisition(mode='ProbabilityImprovement')
    acq_3 = Acquisition(mode='UCB', beta=0.5)
    acq_4 = Acquisition(mode='UCB', beta=1.5)
    acq_list = [acq_1, acq_2, acq_3, acq_4]
    sexp = squaredExponential()
    param = {'x': ('cont', [0, 2 * np.pi])}
    new = True
    colors = ['green', 'red', 'orange', 'black']
    acq_titles = [r'Expected improvement', r'Probability of Improvement', r'GP-UCB $\beta = .5$',
                  r'GP-UCB $\beta = 1.5$']

    for index, acq in enumerate(acq_list):
        np.random.seed(200)
        gp = GaussianProcess(sexp)
        gpgo = GPGO(gp, acq, f, param)
        gpgo._firstRun(n_eval=3)
        plotGPGO(gpgo, param, index=index + 2, new=new)
        new = False

    plt.show()
