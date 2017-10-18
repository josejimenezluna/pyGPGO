#######################################
# pyGPGO examples
# exampleRF: tests the Random Forest surrogate model.
#######################################

import numpy as np
from pyGPGO.surrogates.RandomForest import RandomForest
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Build synthetic data (sine function)
    x = np.arange(0, 2 * np.pi + 0.01, step=np.pi / 8)
    y = np.sin(x)
    X = np.array([np.atleast_2d(u) for u in x])[:, 0]

    rf = RandomForest(n_estimators=20)
    # Fit the model to the data
    rf.fit(X, y)
    # Predict on new data
    xstar = np.arange(0, 2 * np.pi, step=0.01)
    Xstar = np.array([np.atleast_2d(u) for u in xstar])[:, 0]
    ymean, ystd = rf.predict(Xstar, return_std=True)

    # Confidence interval bounds
    lower, upper = ymean - 1.96 * ystd, ymean + 1.96 * ystd

    # Plot values
    plt.figure()
    plt.plot(xstar, ymean, label='Posterior mean')
    plt.plot(xstar, np.sin(xstar), label='True function')
    plt.fill_between(xstar, lower, upper, alpha=0.4, label=r'95% confidence band')
    plt.grid()
    plt.legend(loc=0)
    plt.show()