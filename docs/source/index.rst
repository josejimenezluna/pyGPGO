.. pyGPGO documentation master file, created by
   sphinx-quickstart on Thu Mar 23 17:21:57 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyGPGO: Bayesian Optimization for Python
==================================

pyGPGO is a simple and modular Python (>3.5) package for Bayesian Optimization. It supports:

- Different surrogate models: Gaussian Processes, Student-t Processes, Random Forests, Gradient Boosting Machines.
- Type II Maximum-Likelihood of covariance function hyperparameters.
- MCMC sampling for full-Bayesian inference of hyperparameters (via ``pyMC3``).
- Integrated acquisition functions

Check us out on  `Github <https://github.com/hawk31/pyGPGO>`_.

Overall, pyGPGO is a very easy to use package. In practice, a user needs to specify:

- A function to optimize according to some parameters.
- A dictionary defining parameters, their type and bounds.
- A surrogate model, such as a Gaussian Process, from the surrogates module. Some surrogate models require defining
  a covariance function, with hyperparameters. (from the covfunc module)
- An acquisition strategy, from the acquisition module.
- A GPGO instance, from the GPGO module

A simple example can be checked below::

   import numpy as np
   from pyGPGO.covfunc import squaredExponential
   from pyGPGO.acquisition import Acquisition
   from pyGPGO.surrogates.GaussianProcess import GaussianProcess
   from pyGPGO.GPGO import GPGO

   def f(x):
       return (np.sin(x))


   sexp = squaredExponential()
   gp = GaussianProcess(sexp)
   acq = Acquisition(mode='ExpectedImprovement')
   param = {'x': ('cont', [0, 2 * np.pi])}

   np.random.seed(23)
   gpgo = GPGO(gp, acq, f, param)
   gpgo.run(max_iter=20)


Contents:

.. toctree::
   :maxdepth: 3

   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

