.. pyGPGO documentation master file, created by
   sphinx-quickstart on Thu Mar 23 17:21:57 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyGPGO: Bayesian optimization for Python
==================================

pyGPGO is a simple and modular Python (>3.5) package for Bayesian optimization. It supports:

- Different surrogate models: Gaussian Processes, Student-t Processes, Random Forests, Gradient Boosting Machines.
- Type II Maximum-Likelihood of covariance function hyperparameters.
- MCMC sampling for full-Bayesian inference of hyperparameters (via ``pyMC3``).
- Integrated acquisition functions

Check us out on  `Github <https://github.com/hawk31/pyGPGO>`_.

pyGPGO uses other well-known packages of the Python scientific ecosystem as dependencies:

- numpy
- scipy
- joblib
- scikit-learn
- pyMC3
- theano

These are automatically taken care for in the requirements file.


What is Bayesian Optimization?
==================================

Bayesian optimization is a framework that is useful in several scenarios:

- Your objective function has no closed-form.
- No access to gradients
- In presence of noise
- It may be expensive to evaluate.

The bayesian optimization framework uses a surrogate model to approximate the objective function and chooses to
optimize it according to some acquisition function. This framework gives a lot of freedom to the user in terms
of optimization choices:

- Surrogate model choice
- Covariance function choice
- Acquisition function behaviour
- Hyperparameter treatment

pyGPGO provides an extensive range of choices in each of the previous points, in a modular way. We recommend checking
[Shahriari2016]_ for an in-depth review of the framework if you're interested.


How do I get started with pyGPGO?
==================================

Install the latest stable release from pyPI::

  pip install pyGPGO


or if you're feeling adventurous, install the latest devel version from the Github repository::

  pip install git+https://github.com/hawk31/pyGPGO


pyGPGO is straightforward to use, we only need to specify:

- A function to optimize according to some parameters.
- A dictionary defining parameters, their type and bounds.
- A surrogate model, such as a Gaussian Process, from the ``surrogates`` module. Some surrogate models require defining
  a covariance function, with hyperparameters. (from the ``covfunc`` module)
- An acquisition strategy, from the ``acquisition`` module.
- A GPGO instance, from the ``GPGO`` module

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


There are a couple of tutorials to help get you started on the `tutorials <https://github.com/hawk31/pyGPGO/tree/master/tutorials>`_ folder.

For a full list of features with explanations check our Features section

.. toctree::
   :maxdepth: 1

   features

pyGPGO is not the only package for bayesian optimization in Python, other excellent alternatives exist. For an in-depth comparison
of the features offered by pyGPGO compared to other sofware, check the following section:

.. toctree::
   :maxdepth: 1

   comparison

API documentation
=================

.. toctree::
   :maxdepth: 3

   api


References
==========

.. [Shahriari2016] Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2016). Taking the human out of the loop: A review of Bayesian optimization. Proceedings of the IEEE. http://doi.org/10.1109/JPROC.2015.2494218


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

