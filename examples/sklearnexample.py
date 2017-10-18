#######################################
# pyGPGO examples
# sklearnexample: Optimizes hyperparameters for an SVM classifier 
# on synthetic generated data.
#######################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential


def evaluateModel(C, gamma):
    clf = SVC(C=10**C, gamma=10**gamma)
    return np.average(cross_val_score(clf, X, y))


if __name__ == '__main__':
    np.random.seed(20)
    X, y = make_moons(n_samples=200, noise=0.3)

    cm_bright = ListedColormap(['#fc4349', '#6dbcdb'])

    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.show()

    sexp = squaredExponential()
    gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    acq = Acquisition(mode='UCB', beta=1.5)

    params = {'C':      ('cont', (-4, 5)),
              'gamma':  ('cont', (-4, 5))
             }

    gpgo = GPGO(gp, acq, evaluateModel, params)
    gpgo.run(max_iter=50)
    gpgo.getResult()
