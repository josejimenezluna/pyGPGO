# pyGPGO: Bayesian Optimization for Python
[![Build Status](https://travis-ci.org/hawk31/pyGPGO.svg?branch=master)](https://travis-ci.org/hawk31/pyGPGO)
[![Documentation Status](https://readthedocs.org/projects/pygpgo/badge/?version=latest)](http://pygpgo.readthedocs.io/en/latest/?badge=latest)

![sine](http://i.giphy.com/l3q2s3MQ4bPb5RogU.gif)

pyGPGO is a simple and modular Python (>3.5) package for Bayesian Optimization.

### Installation

Just pip install the repo:


```bash
pip install git+https://github.com/hawk31/pyGPGO
```

### Dependencies

*   Typical Python scientific stuff: `numpy`, `scipy`.
*   `joblib` (Optional, used for parallel computation)
*   `scikit-learn` (Optional, for other surrogates different than GP.)
*   `pyMC3` (Optional, for integrated acquisition functions and MCMC inference)
*   `theano` (Optional) development version. (pyMC3 dependency)

### Features

* Different surrogate models: Gaussian Processes, Student-t Processes, Random Forests, Gradient Boosting Machines.
* Type II Maximum-Likelihood of covariance function hyperparameters. 
* MCMC sampling for full-Bayesian inference of hyperparameters (via `pyMC3`).
* Integrated acquisition functions

### Usage

The user only has to define a function to maximize and a dictionary specifying input space.

```python
c

```

Check the examples folder as well!
