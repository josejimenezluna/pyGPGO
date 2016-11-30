import numpy as np
from covfunc import *
from GPRegressor import GPRegressor
from acquisition import Acquisition
from GPGO import GPGO
from collections import OrderedDict
import matplotlib.pyplot as plt

def rosen(x, y, a = 1, b = 100):
	return ((a - x)**2 + b*(y - x**2)**2)

def rastrigin(x, y, A=10):
	return (2*A + (x**2 - A * np.cos(2 * np.pi *x)) + (y**2 - A * np.cos(2 * np.pi * y)))

def plot_f(x_values, y_values, f):
	z = np.zeros((len(x_values), len(y_values)))
	for i in range(len(x_values)):
		for j in range(len(y_values)):
			z[i, j] = f(x_values[i], y_values[j])
	plt.imshow(z, origin = 'lower')
	plt.colorbar()
	plt.show()

def plot2dgpgo(gpgo, param):
	tested_X = gpgo.GP.X
	n = 100
	r_x, r_y = gpgo.parameter_range[0], gpgo.parameter_range[1]
	x_test = np.linspace(r_x[0], r_x[1], 100)
	y_test = np.linspace(r_y[0], r_y[1], 100)
	z_hat = np.empty((len(x_test), len(y_test)))
	z_var = np.empty((len(x_test), len(y_test)))
	ac = np.empty((len(x_test), len(y_test)))
	for i in range(len(x_test)):
		for j in range(len(y_test)):
			res = gpgo.GP.predict([x_test[i], y_test[j]])
			z_hat[i, j] = res[0]
			z_var[i, j] = res[1][0]
			ac[i, j] = -gpgo._acqWrapper(np.atleast_1d([x_test[i], y_test[j]]))
	plt.figure()
	plt.subplot(221)
	plt.imshow(z_hat.T, origin = 'lower', extent = [-1, 1, -1, 1])
	plt.colorbar()
	plt.plot(tested_X[:, 0], tested_X[:, 1], 'wx', markersize = 10)
	plt.subplot(222)
	plt.imshow(z_var.T, origin = 'lower', extent = [-1, 1, -1, 1])
        plt.plot(tested_X[:, 0], tested_X[:, 1], 'wx', markersize = 10)
	plt.colorbar()
	plt.subplot(223)
	plt.imshow(ac.T, origin = 'lower', extent = [-1, 1, -1, 1])
	plt.colorbar()
	gpgo._optimizeAcq(method = 'L-BFGS-B', n_start = 500)
	plt.plot(gpgo.best[0], gpgo.best[1], 'gx', markersize = 15)
	plt.show()
	
if __name__ == '__main__':
	x = np.linspace(-1, 1, 1000)
	y = np.linspace(-1, 1, 1000)
	plot_f(x, y, rastrigin)
	
	np.random.seed(1337)
	sexp = squaredExponential()
	gp = GPRegressor(sexp)
	acq = Acquisition(mode = 'ExpectedImprovement')
	
	param = OrderedDict()
	param['x'] = ('cont', [-1, 1])
	param['y'] = ('cont', [-1, 1])

	gpgo = GPGO(gp, acq, rastrigin, param)
	#gpgo._sampleParam()
	gpgo._firstRun()
	
	for i in range(5):	
		plot2dgpgo(gpgo, param)
		gpgo.updateGP()

	

	
	
