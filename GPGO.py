import numpy as np
from GPRegressor import GPRegressor
from scipy.optimize import minimize
from cma import fmin
from collections import OrderedDict

class GPGO:
	def __init__(self, GPRegressor, Acquisition, f, parameter_dict):
		self.GP = GPRegressor
		self.A  = Acquisition
		self.f = f
		self.parameters = parameter_dict
		
		self.parameter_key = list(parameter_dict.keys())
		self.parameter_value = list(parameter_dict.values())
		self.parameter_type = [p[0] for p in self.parameter_value]
		self.parameter_range = [p[1] for p in self.parameter_value]

	def _sampleParam(self):
		d = OrderedDict()
		for index, param in enumerate(self.parameter_key):
			if parameter_type[index] == 'int':
				d[param] = np.random.randint(self.parameter_range[index][0], self.parameter_range[index][1])
			elif parameter_type[index] == 'disc':
				d[param] = np.random.choice(self.parameter_range[index])
			elif parameter_type[index] == 'cont':
				d[param] = np.random.uniform(self.parameter_range[index][0], self.parameter_range[index][1])
			else:
				raise ValueError('Unsupported variable type.')
		return d
		
	def _firstRun(self, n_eval = 3):
		self.X = np.empty((n_eval, len(self.parameter_key))
		self.y = np.empty((n_eval, ))
		for i in range(n_eval):
			s_param = self._sampleParam()
			s_param_val = list(s_param.values())
			self.X[i] = s_param_val
			self.y[i] = self.f(**s_param)
		self.GP.fit(X, y)
		self.tau = np.max(self.y)
	
	def _acqWrapper(self, xnew):
		new_mean, new_var = GP.predict(xnew)
		new_std = np.sqrt(new_var)
		return self.A.eval(self.tau, new_mean, new_std)
		
	def _optimizeAcq(self, method = 'L-BFGS-B', nstart = 100):
		start_points = [list(self._sampleParam().values()) for i in range(nstart)]
		x_best = np.empty((n_start, len(self.parameter_key)))
		f_best = np.empty((n_start,))
		#TODO use L-BFGS-B or CMA-ES.
		if method == 'L-BFGS-B':

		elif method = 'CMA-ES': # This can be parallelized
			for index, start_point in enumerate(start_points):
				res = fmin(-self._acqWrapper, start_point)
				x_best[index], f_best[index] = res[0], res[1]
		self.best = x_best[np.argmin(f_best)]		
	
	def updateGP(self):
		self.GP.update(self.best)

	def run(self, max_iter = 10):
		self._firstRun()
		for iteration in range(max_iter):
			self._optimizeAcq()
			self.updateGP()
		
