import numpy as np
from pyGPGO.acquisition import Acquisition


modes = ['ExpectedImprovement', 'ProbabilityImprovement', 'UCB', 'Entropy', 
         'tExpectedImprovement']

modes_mcmc = ['IntegratedExpectedImprovement', 'IntegratedProbabilityImprovement',
              'IntegratedUCB', 'tIntegratedExpectedImprovement']


tau = 1.96
mean = np.array([0])
std = np.array([1])

extra_params = {'beta': 1.5}

means = np.random.randn(1000)
stds = np.random.uniform(0.8, 1.2, 1000)


def test_acq():
    for mode in modes:
        acq = Acquisition(mode=mode)
        acq.eval(tau, mean, std)


def test_acq_mcmc():
    for mode in modes_mcmc:
        acq = Acquisition(mode=mode)
        print(acq.eval(tau, means, stds))


if __name__ == '__main__':
    test_acq()
    test_acq_mcmc()
