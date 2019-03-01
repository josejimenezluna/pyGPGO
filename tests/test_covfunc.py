import numpy as np
from pyGPGO.covfunc import squaredExponential, matern, matern32, matern52, \
                           gammaExponential, rationalQuadratic, expSine, dotProd


covfuncs = [squaredExponential(), matern(), matern32(), matern52(), gammaExponential(),
            rationalQuadratic(), expSine(), dotProd()]

grad_enabled = [squaredExponential(), matern32(), matern52(), gammaExponential(),
                rationalQuadratic(), expSine()]

# Some kernels do not have gradient computation enabled, such is the case 
# of the generalised matérn kernel.
#
# All (but the dotProd kernel) have a characteristic length-scale l that
# we test for here.

covariance_classes = dict(squaredExponential=squaredExponential, matern=matern, matern32=matern32, matern52=matern52,
                          gammaExponential=gammaExponential, rationalQuadratic=rationalQuadratic, dotProd=dotProd)

hyperparameters_interval = dict(squaredExponential=dict(l=(0, 2.0), sigmaf=(0, 0.5), sigman=(0, 0.5)),
                                matern=dict(l=(0, 2.0), sigmaf=(0, 0.5), sigman=(0, 0.5)),
                                matern32=dict(l=(0, 2.0), sigmaf=(0, 0.5), sigman=(0, 0.5)),
                                matern52=dict(l=(0, 2.0), sigmaf=(0, 0.5), sigman=(0, 0.5)),
                                gammaExponential=dict(gamma=(0,2.0), l=(0, 2.0), sigmaf=(0, 0.5), sigman=(0, 0.5)),
                                rationalQuadratic=dict(alpha=(0,2.0), l=(0, 2.0), sigmaf=(0, 0.5), sigman=(0, 0.5)),
                                dotProd=dict(sigmaf=(0, 0.5), sigman=(0, 0.5)))

def generate_hyperparameters(**hyperparmeter_interval):
    generated_hyperparameters = dict()
    for hyperparameter, bound in hyperparmeter_interval.items():
        generated_hyperparameters[hyperparameter] = np.random.uniform(bound[0], bound[1])
    return generated_hyperparameters


def test_psd_covfunc():
    # Check if generated covariance functions are positive definite
    np.random.seed(0)
    for name in covariance_classes:
        for i in range(10):
            generated_hyperparameters = generate_hyperparameters(**hyperparameters_interval[name])
            cov = covariance_classes[name](**generated_hyperparameters)
            for j in range(100):
                X = np.random.randn(10, 2)
                eigvals = np.linalg.eigvals(cov.K(X,X))
                assert (eigvals > 0).all()


def test_sim():
    rng = np.random.RandomState(0)
    X = np.random.randn(100, 3)
    for cov in covfuncs:
        cov.K(X, X)


def test_grad(): 
    rng = np.random.RandomState(0)
    X = np.random.randn(3, 3)
    for cov in grad_enabled:
        cov.gradK(X, X, 'l')


if __name__ == '__main__':
    test_sim()
    test_grad()