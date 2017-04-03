import numpy as np
from scipy.linalg import cholesky, inv
from numpy.random import multivariate_normal, uniform
from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess

rng = np.random.RandomState(0)
X = rng.uniform(0, 5, 20)[:, np.newaxis]
y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

def computeStuff(l, scale=20):
    S_theta = np.eye(X.shape[0])
    sexp = squaredExponential(l=l)
    gp = GaussianProcess(sexp)
    Sigma_theta = gp.covfunc.K(X, X)
    R_theta = S_theta - np.dot(np.dot(S_theta, inv(S_theta + Sigma_theta)), S_theta)
    L_theta = cholesky(R_theta, lower=True)
    m_theta = np.dot(R_theta.dot(inv(S_theta)), g)
    return S_theta, Sigma_theta, R_theta, L_theta,m_theta



def sliceSampling():
    l = 1
    S_theta, Sigma_theta, R_theta, L_theta, m_theta = computeStuff(l = l)
    # Draw surrogate data
    g = multivariate_normal(y, S_theta)
    nu = np.dot(inv(L_theta), (y - m_theta))
    v = uniform(0, 1)
    l_min = l - v
    l_max = l_min + scale
    u = uniform(0, 1)
    # Determine threshold
    prior = scipy.stats.gamma.pdf(l, 2, 1)
    y = u * np.exp(gp.logp) * scipy.stats.multivariate_normal.pdf(g, np.zeros(X.shape[0]), Sigma_theta + S_theta) * prior
    l_tilde = uniform(l_min, l_max)
    _, _, Rtilde, Ltilde, m_tilde = computeStuff(l_tilde)
    f_tilde = np.dot(Ltilde, nu) + m_tilde


sexp = squaredExponential()
K = sexp.K(X, X)
L = cholesky(K, lower=True)

