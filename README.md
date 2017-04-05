# pyGPGO: A simple Python package for Bayesian Optimization
[![Build Status](https://travis-ci.org/hawk31/pyGPGO.svg?branch=master)](https://travis-ci.org/hawk31/pyGPGO)
[![Documentation Status](https://readthedocs.org/projects/pygpgo/badge/?version=latest)](http://pygpgo.readthedocs.io/en/latest/?badge=latest)

![sine](http://i.giphy.com/l3q2s3MQ4bPb5RogU.gif)

pyGPGO is a simple Python (>3.5) package for Bayesian Optimization using Gaussian Process as surrogate model.

### Installation

Available in PyPI

```bash
pip install pyGPGO
```

Recommended: development version

```bash
pip install git+https://github.com/hawk31/pyGPGO
```

### Dependencies

*   Typical Python scientific stuff: `numpy`, `scipy`.
*   `joblib` (Optional, used for parallelizing computation)
*   `pyMC3` (Optional, for integrated acquisition functions and MCMC inference)
*   `theano` (Optional) development version. (pyMC3 dependency)
### Usage

The user only has to define a function and a dictionary.

```python
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

```
