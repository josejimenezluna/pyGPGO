from collections import OrderedDict

import numpy as np
from cma import fmin
from scipy.optimize import minimize
from joblib import Parallel, delayed


class GPGO:
    def __init__(self, GPRegressor, Acquisition, f, parameter_dict, n_jobs = 1):
        self.GP = GPRegressor
        self.A = Acquisition
        self.f = f
        self.parameters = parameter_dict
        self.n_jobs = n_jobs

        self.parameter_key = list(parameter_dict.keys())
        self.parameter_value = list(parameter_dict.values())
        self.parameter_type = [p[0] for p in self.parameter_value]
        self.parameter_range = [p[1] for p in self.parameter_value]

        self.history = []

    def _sampleParam(self):
        d = OrderedDict()
        for index, param in enumerate(self.parameter_key):
            if self.parameter_type[index] == 'int':
                d[param] = np.random.randint(self.parameter_range[index][0], self.parameter_range[index][1])
            elif self.parameter_type[index] == 'cont':
                d[param] = np.random.uniform(self.parameter_range[index][0], self.parameter_range[index][1])
            else:
                raise ValueError('Unsupported variable type.')
        return d

    def _firstRun(self, n_eval=3):
        self.X = np.empty((n_eval, len(self.parameter_key)))
        self.y = np.empty((n_eval,))
        for i in range(n_eval):
            s_param = self._sampleParam()
            s_param_val = list(s_param.values())
            self.X[i] = s_param_val
            self.y[i] = self.f(**s_param)
        self.GP.fit(self.X, self.y)
        self.tau = np.max(self.y)
        self.history.append(self.tau)

    def _acqWrapper(self, xnew):  # Returns minimum for optimization purposes
        new_mean, new_var = self.GP.predict(xnew, return_std=True)
        new_std = np.sqrt(new_var)
        return -self.A.eval(self.tau, new_mean, new_std)

    def _optimizeAcq(self, method='L-BFGS-B', n_start=100):
        start_points_dict = [self._sampleParam() for i in range(n_start)]
        start_points_arr = np.array([list(s.values()) for s in start_points_dict])
        x_best = np.empty((n_start, len(self.parameter_key)))
        f_best = np.empty((n_start,))
        if method == 'L-BFGS-B':
            if self.n_jobs == 1:
                for index, start_point in enumerate(start_points_arr):
                    res = minimize(self._acqWrapper, x0=start_point, method=method,
                                   bounds=self.parameter_range)
                    x_best[index], f_best[index] = res.x, res.fun[0]
            else:
                opt = Parallel(n_jobs=self.n_jobs)(delayed(minimize)(self._acqWrapper,
                                                                     x0 = start_point,
                                                                     method = 'L-BFGS-B',
                                                                     bounds = self.parameter_range) for start_point in start_points_arr)
                x_best = np.array([res.x for res in opt])
                f_best = np.array([res.fun[0] for res in opt])

        elif method == 'CMA-ES':
            for index, start_point in enumerate(start_points_arr):
                res = fmin(self._acqWrapper, x0=start_point, sigma0=0.1)
                x_best[index], f_best[index] = res[0], res[1]
        self.best = x_best[np.argmin(f_best)]

    def updateGP(self):
        kw = {param: self.best[i] for i, param in enumerate(self.parameter_key)}
        f_new = self.f(**kw)
        self.GP.update(np.atleast_2d(self.best), np.atleast_1d(f_new))
        self.tau = np.max(self.GP.y)
        self.history.append(self.tau)

    def run(self, max_iter=10, init_evals = 3):
        self.init_evals = init_evals
        self._firstRun(self.init_evals)
        for iteration in range(max_iter):
            self._optimizeAcq()
            self.updateGP()
