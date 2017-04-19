#WIP
import numpy as np
import scipy as sp
import theano
import theano.tensor as tt
import theano.tensor.nlinalg
import pymc3 as pm
from pyGPGO.covfunc import squaredExponential, matern
from pyGPGO.surrogates.tStudentProcess import tStudentProcess
import matplotlib.pyplot as plt

covariance_equivalence = {'squaredExponential': pm.gp.cov.ExpQuad,
                          'matern52': pm.gp.cov.Matern52,
                          'matern32': pm.gp.cov.Matern32}


class tStudentProcessMCMC:
    def __init__(self, covfunc, nu=3.0, niter=2000, burnin=1000, init='ADVI', step=None):
        self.covfunc = covfunc
        self.nu = nu
        self.niter = niter
        self.burnin = burnin
        self.init = init
        self.step = step

    def _extractParam(self, unittrace, covparams):
        d = {}
        for key, value in unittrace.items():
            if key in covparams:
                d[key] = value
        if 'v' in covparams:
            d['v'] = 5 / 2
        return d

    def fit(self, X, y):
        self.X = X
        self.n = self.X.shape[0]
        self.y = y
        self.model = pm.Model()

        with self.model as model:
            l = pm.Uniform('l', 0, 10)

            log_s2_f = pm.Uniform('log_s2_f', lower=-7, upper=5)
            s2_f = pm.Deterministic('sigmaf', tt.exp(log_s2_f))

            log_s2_n = pm.Uniform('log_s2_n', lower=-7, upper=5)
            s2_n = pm.Deterministic('sigman', tt.exp(log_s2_n))

            f_cov = s2_f * covariance_equivalence[type(self.covfunc).__name__](1, l)
            Sigma = f_cov(self.X) + tt.eye(self.n) * s2_n ** 2
            y_obs = pm.MvStudentT('y_obs', nu=self.nu, mu=np.zeros(self.n), Sigma=Sigma, observed=self.y)
        with self.model as model:
            if self.step is not None:
                self.trace = pm.sample(self.niter, step=self.step())[self.burnin:]
            else:
                self.trace = pm.sample(self.niter, init=self.init)[self.burnin:]

    def predict(self, Xstar, return_std=False, nsamples=100):
        chunk = list(self.trace)
        chunk = chunk[::-1][:nsamples]
        post_mean = []
        post_var = []
        for posterior_sample in chunk:
            params = self._extractParam(posterior_sample, self.covfunc.parameters)
            covfunc = self.covfunc.__class__(**params)
            gp = tStudentProcess(covfunc, nu=self.nu + self.n)
            gp.fit(self.X, self.y)
            m, s = gp.predict(Xstar, return_std=return_std)
            post_mean.append(m)
            post_var.append(s)
        return np.array(post_mean), np.array(post_var)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from pyGPGO.acquisition import Acquisition
    from pyGPGO.covfunc import squaredExponential
    from pyGPGO.GPGO import GPGO

    if __name__ == '__main__':
        sexp = squaredExponential()
        gp = tStudentProcessMCMC(sexp, step=pm.Slice)

        def f(x):
            return np.sin(x)

        np.random.seed(200)
        param = {'x': ('cont', [0, 6])}
        acq = Acquisition(mode='IntegratedExpectedImprovement')
        gpgo = GPGO(gp, acq, f, param)
        gpgo._firstRun(n_eval=7)

        plt.figure()
        plt.subplot(2, 1, 1)

        Z = np.linspace(0, 6, 100)[:, None]
        post_mean, post_var = gpgo.GP.predict(Z, return_std=True, nsamples=200)
        for i in range(200):
            plt.plot(Z.flatten(), post_mean[i], linewidth=0.4)

        plt.plot(gpgo.GP.X.flatten(), gpgo.GP.y, 'X', label='Sampled data', markersize=10, color='red')
        plt.grid()
        plt.legend()

        xtest = np.linspace(0, 6, 200)[:, np.newaxis]
        a = [-gpgo._acqWrapper(np.atleast_2d(x)) for x in xtest]
        plt.subplot(2, 1, 2)
        plt.plot(xtest, a, label='Integrated Expected Improvement')
        plt.grid()
        plt.legend()
        plt.show()