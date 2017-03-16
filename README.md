# pyGPGO

![sine](http://i.giphy.com/l3q2s3MQ4bPb5RogU.gif)

pyGPGO is a simple Python (>3.5) package for Bayesian Optimization using Gaussian Process as surrogate model.

### Installation

Available in PyPI

```bash
pip install pyGPGO
```

### Dependencies

At the moment, the only dependency is `numpy` and `joblib`.

### Usage

The user only has to define a function and a dictionary.

```python
import numpy as np
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.GPRegressor import GPRegressor
from pyGPGO.GPGO import GPGO

def f(x):
    return (np.sin(x))


sexp = squaredExponential()
gp = GPRegressor(sexp, sigma = 1e-8)
acq = Acquisition(mode='ExpectedImprovement')
param = {'x': ('cont', [0, 2 * np.pi])}

np.random.seed(23)
gpgo = GPGO(gp, acq, f, param)
gpgo.run(max_iter=20)

```
