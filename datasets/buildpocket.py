import numpy as np

def buildpocket():
    X = np.load('/shared/jose/open/data/pocket_training/X_pocket_training.npy')
    y = np.load('/shared/jose/open/data/pocket_training/y_pocket_training.npy')
    e = np.average(X < 10e-4, axis = (2, 3, 4))
    e = np.hstack((np.atleast_2d(y).T, e))
    np.savetxt('/home/jose/pyGPGO/datasets/pocket.csv', X = e, delimiter = ',')
