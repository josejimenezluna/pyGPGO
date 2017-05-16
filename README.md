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

Optionally, install `pyMC3`

```bash
git clone https://github.com/pymc-devs/pymc3
cd pymc3
pip install -r requirements.txt
python setup.py install
```

### Dependencies

*   Typical Python scientific stuff: `numpy`, `scipy`.
*   `joblib` (Optional, used for parallel computation)
*   `scikit-learn` (Optional, for other surrogates different than GP.)
*   `pyMC3` (Optional, for integrated acquisition functions and MCMC inference)
*   `theano` (Optional) development version. (pyMC3 dependency)

All dependencies except `pyMC3` are taken care for in the requirements file.

### Features

* Different surrogate models: Gaussian Processes, Student-t Processes, Random Forests, Gradient Boosting Machines.
* Type II Maximum-Likelihood of covariance function hyperparameters. 
* MCMC sampling for full-Bayesian inference of hyperparameters (via `pyMC3`).
* Integrated acquisition functions

### Usage

The user only has to define a function to maximize and a dictionary specifying input space.

```python
import numpy as np
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO


def f(x, y):
    # Franke's function (https://www.mathworks.com/help/curvefit/franke.html)
    one = 0.75 * np.exp(-(9*x-2)**2/4 - (9*y - 2)**2/4)
    two = 0.75 * np.exp(-(9*x+1)**2/49 - (9*y + 1)/10)
    three = 0.5 * np.exp(-(9*x - 7)**2/4 - (9*y -3)**2/4)
    four = 0.25 * np.exp(-(9*x -4)**2 - (9*y-7)**2)
    return one + two + three - four

sexp = matern32()
gp = tStudentProcess(sexp)
acq = Acquisition(mode='ExpectedImprovement')
param = {'x': ('cont', [0, 1]),
         'y': ('cont', [0, 1])}

np.random.seed(1337)
gpgo = GPGO(gp, acq, f, param)
gpgo.run(max_iter=10)

```

Check the examples folder as well!
