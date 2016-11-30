import numpy as np
import matplotlib.pyplot as plt
from GPRegressor import GPRegressor
from acquisition import Acquisition
from covfunc import *
from GPGO import GPGO

def plotGPGO(gpgo, param):
    param_value = list(param.values())[0][1]
    x_test = np.linspace(param_value[0], param_value[1], 1000).reshape((1000, 1))
    hat = gp.predict(x_test, return_std = True)
    y_hat, y_var = hat[0], hat[1]
    l, u = y_hat - 1.96 * np.sqrt(y_var), y_hat + 1.96 * np.sqrt(y_var)
    plt.figure()
    plt.subplot(211)
    plt.fill_between(x_test.flatten(), l, u, alpha = 0.2)
    plt.show()
    a = np.array([-gpgo._acqWrapper(np.atleast_1d(x)) for x in x_test]).flatten()
    plt.subplot(212)
    plt.plot(x_test, a, color = 'green')
    gpgo._optimizeAcq(method = 'L-BFGS-B', n_start = 1000)
    plt.axvline(x = gpgo.best)
    plt.show()

if __name__ == '__main__':
    np.random.seed(321)
    def f(x):
	return(np.sin(x))

    sexp = squaredExponential()
    gp = GPRegressor(sexp)
    acq = Acquisition(mode = 'ExpectedImprovement')
    param = {'x': ('cont', [0, 2 * np.pi])}

    gpgo = GPGO(gp, acq, f, param)
    gpgo._firstRun()
    
    for i in range(5):
	plotGPGO(gpgo, param)
        gpgo.updateGP() 
