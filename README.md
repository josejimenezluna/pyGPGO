# pyGPGO

![sine](http://imgur.com/KI7i3mI)

pyGPGO is a simple Python (>3.5) package for Bayesian Optimization using Gaussian Process as surrogate model.

### Dependencies

At the moment, the only dependency is `numpy`. The Anaconda Intel MKL is recommended.

### Usage

_Under construction_

The user only has to define a function and a dictionary.

```python
from covfunc import squaredExponential
from acquisition import Acquisition
from GPRegressor import GPRegressor
from GPGO import GPGO

def f(x):
	return (np.sin(x))


sexp = squaredExponential()
gp = GPRegressor(sexp)
acq = Acquisition(mode='ExpectedImprovement')
param = {'x': ('cont', [0, 2 * np.pi])}

gpgo = GPGO(gp, acq, f, param)
gpgo.run(max_iter=20)

```
