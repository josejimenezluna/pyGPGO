import numpy as np

class SimulatedAnnealing:
	def __init__(self, sampler, f, T = 100, cooling = 0.9):
		self.sampler = sampler
		self.f = f
		self.T = T
		self.cooling = cooling
		self.history = []
		self.best = -np.inf

	def _accept(self, sample):
		newval = f(**sample)
		if newval > self.best:
			self.best = newval
		else:
			p = np.exp(-(newval - self.best) / self.T) 
			if np.random.uniform() < p:
				self.best = newval
		self.history.append(self.best)
		self.T = self.cooling * self.T

	def run(self, n_iter = 50):
		for i in range(n_iter):
			sample = self.sampler()
			self._accept(sample)
		 
		
