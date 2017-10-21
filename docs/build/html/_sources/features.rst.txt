Features
==================

The Bayesian optimization framework is very flexible, as it allows for choices in many
steps of its design. To name a few of the choices that pyGPGO provides to the user:

Surrogate models :class:`pyGPGO.surrogates`
----------------

The framework works by specifying a model that will approximate our target function, 
better after each evaluation. The most common surrogate in the literature is the Gaussian
Process, but the framework is model agnostic. Some featured models are:

- Gaussian Processes (:class:`pyGPGO.surrogates.GaussianProcess` :class:`pyGPGO.surrogates.GaussianProcessMCMC`): By far the most common choice, it needs the user to specify a covariance function (detailed in the next section), measuring similarity among training examples. For a good introduction to Gaussian Processes, check [@Rasmussen-Williams2004].
- Student-t Processes (:class:`pyGPGO.surrogates.tStudentProcess` :class:`pyGPGO.surrogates.tStudentProcessMCMC`): Some functions benefit from the heavy-tailed nature of the Student-t distribution. It also requires providing a covariance function.
- Random Forests (:class:`pyGPGO.surrogates.RandomForest`): provided by `sklearn`, it represents a nonparametric surrogate model. Does not require specifying a covariance function. A class for Extra Random Forests is also available. Posterior variance is approximated by averaging the variance of each subtree [@reference].
- Gradient Boosting Machines (:class:`pyGPGO.surrogates.BoostedTrees`): similar to the latter, posterior variance is approximated using quantile regression.


Covariance functions :class:`pyGPGO.covfunc`
--------------------

These determine how similar training examples are for the surrogate model. Most of these also 
have hyperparameters that need to be taken into account. pyGPGO implements
the most common covariance functions and its gradients w.r.t. hyperparamers,
that we briefly list here.

- Squared Exponential
- Matérn
- Gamma-Exponential
- Rational-Quadratic
- ArcSine
- Dot-product


Acquisition behaviour :class:`pyGPGO.acquisition`
---------------------

In each iteration of the framework, we choose the next point to evaluate according to a behaviour,
dictated by what we call an acquisition function, leveraging exploration and exploitation of
the sampled space. pyGPGO supports the most common acquisition functions in the literature.

- Probability of improvement: chooses the next point according to the probability of improvement w.r.t. the best observed value.
- Expected improvement: similar to probability of improvement, also weighes the probability by the amount improved. Naturally balances exploration and exploitation and is by far the most used acquisition function in the literature.
- Upper confidence limit: Features a beta parameter to explicitly control the balance of exploration vs exploitation. Higher beta values would higher levels of exploration.
- Entropy: Information-theory based acquisition function.

Integrated version of these are also available for the MCMC sampling versions of surrogate
models.

Hyperparameter treatment
------------------------

Covariance functions also have hyperparameters, and their treatment is also thoroughly discussed in the literature (see [@Shahriari2016]).
To summarize, we mainly have two options available:

- Optimizing the marginal log-likelihood, also called the Empirical Bayes approach. pyGPGO supports this feature using analytical gradients for almost all acquisition functions.
- The full Bayesian approach takes into account the uncertainty caused by the hyperparameters in the optimization procedure by marginalizing them, thatis, integrating over them. pyGPGO implements this via MCMC sampling provided by the pyMC3 software, which in turns also provides an easy way for the user to choose whatever sampler they wish.

References
----------

[@Rasmussen-Williams2004]: Rasmussen, C. E., & Williams, C. K. I. (2004). Gaussian processes for machine learning. International journal of neural systems (Vol. 14). http://doi.org/10.1142/S0129065704001899

[@Shahriari2016]: Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2016). Taking the human out of the loop: A review of Bayesian optimization. Proceedings of the IEEE. http://doi.org/10.1109/JPROC.2015.2494218
