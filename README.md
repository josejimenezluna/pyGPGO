# pyGPGO: Bayesian Optimization for Python
[![Build Status](https://travis-ci.org/hawk31/pyGPGO.svg?branch=master)](https://travis-ci.org/hawk31/pyGPGO)
[![codecov](https://codecov.io/gh/hawk31/pyGPGO/branch/master/graph/badge.svg)](https://codecov.io/gh/hawk31/pyGPGO)
[![Documentation Status](https://readthedocs.org/projects/pygpgo/badge/?version=latest)](http://pygpgo.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/74589922.svg)](https://zenodo.org/badge/latestdoi/74589922)
[![status](http://joss.theoj.org/papers/7d60820fabf7fa81501e3d638cac522d/status.svg)](http://joss.theoj.org/papers/7d60820fabf7fa81501e3d638cac522d)


![sine](http://i.giphy.com/l3q2s3MQ4bPb5RogU.gif)

## Now with Ensemble Goodness Added

pyGPGO is a simple and modular Python (>3.5) package for bayesian optimization. 

Bayesian optimization is a framework that can be used in situations where:

* Your objective function may not have a closed-form. (e.g. the result of a simulation)
* No gradient information is available.
* Function evaluations may be noisy.
* Evaluations are expensive (time/cost-wise)
  
This fork adds the ability to ensemble Gaussian Processes or even Random Forests and Boosted Trees
as the surrogate function.

* Should reduce the possibility of model regret for expensive test runs.
* Reduce model variance and improve prediction should lead to improved optimization.


### Installation

Retrieve our fork from the repo,

```bash
pip install git+https://github.com/dataronio/pyGPGO
```

Check our documentation in http://pygpgo.readthedocs.io/.



### Dependencies

*   Typical Python scientific stuff: `numpy`, `scipy`.
*   `joblib`, used for parallel computation
*   `scikit-learn`, for other surrogates different from Gaussian Processes.
*   `pyMC3`, for integrated acquisition functions and MCMC inference.
*   `theano` for GPU processing.

All dependencies are automatically taken care for in the requirements file.

### Features

* Different surrogate models: Gaussian Processes, Student-t Processes, Random Forests, Gradient Boosting Machines.
* Type II Maximum-Likelihood of covariance function hyperparameters. 
* MCMC sampling for full-Bayesian inference of hyperparameters (via `pyMC3`).
* Integrated acquisition functions
  
### An Example of Using an Ensemble Gaussian Process!

The user only has to define a function to maximize and a dictionary specifying input space.

```python
import numpy as np
from pyGPGO.covfunc import matern32
from pyGPGO.covfunc import matern52
from pyGPGO.covfunc import gammaExponential
from pyGPGO.covfunc import expSine
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.Ensemble import Ensemble
from pyGPGO.covfunc import squaredExponential
from pyGPGO.covfunc import rationalQuadratic


def f(x, y):
    # Franke's function (https://www.mathworks.com/help/curvefit/franke.html)
    one = 0.75 * np.exp(-(9 * x - 2) ** 2 / 4 - (9 * y - 2) ** 2 / 4)
    two = 0.75 * np.exp(-(9 * x + 1) ** 2/ 49 - (9 * y + 1) / 10)
    three = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4 - (9 * y -3) ** 2 / 4)
    four = 0.25 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return one + two + three - four

np.random.seed(42)
mat32 = matern32()
mat52 = matern52()
gamex = gammaExponential()
sqex = squaredExponential()
rquad = rationalQuadratic()
gpmat32 = GaussianProcess(mat32,optimize=False, usegrads=False)
gpmat52 = GaussianProcess(mat52,optimize=False, usegrads=False)
gpsqex = GaussianProcess(sqex,optimize=False)
gprquad = GaussianProcess(rquad,optimize=False)
gpgamex = GaussianProcess(gamex,optimize=False)
acqEI = Acquisition(mode='ExpectedImprovement')
param = {'x': ('cont', [0, 1]),
         'y': ('cont', [0, 1])}
ens = Ensemble([gprquad,gpsqex,gpmat32,gpgamex,gpmat52])
gpgo = GPGO(ens, acqEI, franke, param)
gpgo.run(max_iter=100)

```

Check the `tutorials` and `examples` folders for more ideas on how to use the software.

### Citation

If you use pyGPGO in academic work please cite:

Jiménez, J., & Ginebra, J. (2017). pyGPGO: Bayesian Optimization for Python. The Journal of Open Source Software, 2, 431.
