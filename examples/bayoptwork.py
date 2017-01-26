import numpy as np
from GPRegressor import GPRegressor
from covfunc import squaredExponential
import matplotlib.pyplot as plt


if __name__ == '__main__':
	# Build synthetic data (sine function)
	x = np.arange(0, 2 * np.pi + 0.01, step = np.pi / 1.5)
	y = np.sin(x)
	X = np.array([np.atleast_2d(u) for u in x])[:, 0]
	
	# Specify covariance function
	sexp = squaredExponential()
	# Instantiate GPRegressor class
	gp = GPRegressor(sexp)
	# Fit the model to the data
	gp.fit(X, y)
	
	# Predict on new data
	xstar = np.arange(0, 2 * np.pi, step = 0.01)
	Xstar = np.array([np.atleast_2d(u) for u in xstar])[:, 0]
	ymean, ystd = gp.predict(Xstar, return_std = True)
	
	# Confidence interval bounds
	lower, upper = ymean - 1.96 * ystd, ymean + 1.96 * ystd
	
	# Plot values
	plt.figure()
	plt.plot(xstar, ymean, label = 'Posterior mean')
	#plt.plot(xstar, np.sin(xstar), label = 'True function')
	#plt.fill_between(xstar, lower, upper, alpha = 0.4, label = '95% confidence band')
	plt.plot(xstar, lower, '--', label = 'Lower confidence bound')
	plt.plot(xstar, upper, '--', label = 'Upper confidence bound')
	plt.axhline(y = np.max(lower), color = 'black')
	plt.axvspan(0, .68, color = 'grey', alpha = 0.3)
	plt.plot(xstar[np.argmax(lower)], np.max(lower), '*', markersize = 20)
	plt.axvspan(3.04, 7, color = 'grey', alpha = 0.3, label = 'Discarded region')
	plt.text(3.75, 0.75, 'max LCB')
	plt.grid()
	plt.legend(loc = 0)
	plt.show()
	
