import numpy as np
from scipy.stats import norm


# TODO Reimplement fs to accept GP, not mean/std

class Acquisition:
    def __init__(self, mode, eps=1e-04, **params):
        self.params = params
        self.eps = eps
        if mode == 'ExpectedImprovement':
            self.f = self.ExpectedImprovement
        elif mode == 'ProbabilityImprovement':
            self.f = self.ProbabilityImprovement
        elif mode == 'UCB':
            self.f = self.UCB
        else:
            raise NotImplementedError('Not recognised acquisition function')

    def ProbabilityImprovement(self, tau, mean, std):
        z = (mean - tau - self.eps) / std
        return norm.cdf(z)

    def ExpectedImprovement(self, tau, mean, std):
        z = (mean - tau - self.eps) / std
        return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)[0]

    def UCB(self, tau, mean, std, beta):
        return mean + beta * std

    def Greedy(self, tau, mean, std):
        return mean

    def eval(self, tau, mean, std):
        return self.f(tau, mean, std, **self.params)
