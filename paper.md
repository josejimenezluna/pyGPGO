---
title: 'pyGPGO: Bayesian Optimization for Python'
tags:
  - machine-learning
  - bayesian
  - optimization
authors:
 - name: José Jiménez
   orcid: 0000-0002-5335-7834
   affiliation: 1
 - name: Josep Ginebra
   orcid: 0000-0001-9521-9635
   affiliation: 2
affiliations:
 - name: Computational Biophysics Laboratory. Parc de Recerca Biomèdica de Barcelona (PRBB). 
         Universitat Pompeu Fabra (UPF).
   index: 1
 - name: Department of Statistics and Operations Research.
         Universitat Politècnica de Catalunya (UPC)-
   index: 2
date: 4 September 2017
bibliography: paper.bib
---

# Summary

Bayesian optimization has risen over the last few years as a very attractive method to optimize
expensive to evaluate, black box, derivative-free and possibly noisy functions [@Shahriari2016,
@Snoek2012]. It has grabbed the attention of machine learning researchers over simpler model hyperparameter optimization strategies, such as grid search or random search [@Bergstra2012]. The Bayesian optimization
framework uses prior information and evidence to define a posterior distribution over the space of functions.

*pyGPGO* is an open-source Python package for Bayesian Optimization. This framework is 
inherently modular, as there are many design choices, such as surrogate model choice, 
covariance function specification or acquisition function behaviour or hyperparameter
treatment, to name a few. While other software for Bayesian Optimization exists, either they
are restricted to non-commercial applications [@SpearmintSnoek2012] or are not modular nor extensive
enough to accomodate the framework's flexibility [@scikitoptimize, @yelpmoe]. *pyGPGO* aims to provide
a wide range of choices, such as:

- Different surrogate models: Gaussian Processes, Student-t Processes, Random Forests (& variants)
  and Gradient Boosting Machines.
- Most usual covariance function structures, as well as their derivatives: squared exponential,
  Matèrn, gamma-exponential, rational-quadratic, exponential-sine and dot-product kernel.
- Several acquisition function behaviours: probability of improvement, expected improvement,
  upper confidence bound and entropy-based, as well as their integrated versions.
- Type II maximum-likelihood estimation of covariance hyperparameters.
- MCMC sampling for the full-bayesian treatment of hyperparameters (via `pyMC3`)


*pyGPGO* is MIT-licensed and can be retrieved from both [GitHub](https://github.com/hawk31/pyGPGO)
and [PyPI](https://pypi.python.org/pypi/pyGPGO/0.3.0.dev1), with extensive documentation available at [ReadTheDocs](http://pygpgo.readthedocs.io/en/latest/).

![pyGPGO in action.](franke.gif)


# Future work

*pyGPGO* is an ongoing project, and as such there are several improvements that will be tackled
in the near future:

- Support for linear combinations of covariance functions, with automatic gradient computation.
- Support for more diverse acquisition functions, such as Predictive Entropy Search [@Hernandez-Lobato2014].
- A constrained Bayesian Optimization class is planned for the near future. [@Gardner2014]
 

# References
