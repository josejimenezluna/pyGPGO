import os
import numpy as np
from GPGO import GPGO
from GPRegressor import GPRegressor
from acquisition import Acquisition
from covfunc import squaredExponential
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
from testing.other_go import SimulatedAnnealing
from testing.modaux import SVM, RF, KNN, MLP


class loss:
    def __init__(self, model, X, y, method='holdout', problem='binary'):
        self.model = model
        self.X = X
        self.y = y
        self.method = method
        self.problem = problem
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        if self.problem == 'binary':
            self.loss = log_loss
        elif self.problem == 'cont':
            self.loss = mean_squared_error
        else:
            self.loss = log_loss

    def evaluateLoss(self, **param):
        if self.method == 'holdout':
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=93)
            clf = self.model.__class__(**param).eval()
            clf.fit(X_train, y_train)
            if self.problem == 'binary':
                yhat = clf.predict_proba(X_test)[:, 1]
            elif self.problem == 'cont':
                yhat = clf.predict(X_test)
            else:
                yhat = clf.predict_proba(X_test)
            return (- self.loss(y_test, yhat))
        elif self.method == '5fold':
            kf = KFold(n_splits=5, shuffle=True, random_state=93)
            losses = []
            for train_index, test_index in kf.split(self.X):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                clf = self.model.__class__(**param).eval()
                clf.fit(X_train, y_train)
                if self.problem == 'binary':
                    yhat = clf.predict_proba(X_test)[:, 1]
                elif self.problem == 'cont':
                    yhat = clf.predict(X_test)
                else:
                    yhat = clf.predict_proba(X_test)
                losses.append(- self.loss(y_test, yhat))
            return (np.average(losses))


def cumMax(history):
    n = len(history)
    res = np.empty((n,))
    for i in range(n):
        res[i] = np.max(history[:(i + 1)])
    return (res)


def build(csv_path, target_index, header=None):
    data = pd.read_csv(csv_path, header=header)
    data = data.as_matrix()
    y = data[:, target_index]
    X = np.delete(data, obj=np.array([target_index]), axis=1)
    return X, y


# def evaluateDataset(csv_path, target_index, method='holdout', seed=230, max_iter=50):
#     print('Now evaluating {}...'.format(csv_path))
#     X, y = build(csv_path, target_index)
#
#     clf = SVM()
#     wrapper = loss(clf, X, y, method=method)
#
#     np.random.seed(seed)
#     sexp = squaredExponential()
#     gp = GPRegressor(sexp)
#     acq = Acquisition(mode='ExpectedImprovement')
#     parameter_dict = {'C': ('cont', (0.001, 20)), 'gamma': ('cont', (0.001, 20))}
#     gpgo = GPGO(gp, acq, wrapper.evaluateLoss, parameter_dict)
#     gpgo.run(max_iter=max_iter)
#
#     np.random.seed(seed)
#     r = evaluateRandom(gpgo, wrapper, n_eval=max_iter + 1)
#     r = cumMax(r)
#
#     np.random.seed(seed)
#     sa = evaluateSA(gpgo, wrapper, n_eval=max_iter + 1)
#     sa = cumMax(sa)
#
#     return np.array(gpgo.history), r, sa

def evaluateDataset(csv_path, target_index, model, parameter_dict, method='holdout', seed=230, max_iter=50):
    print('Now evaluating {}...'.format(csv_path))
    X, y = build(csv_path, target_index)

    wrapper = loss(model, X, y, method=method)

    np.random.seed(seed)
    sexp = squaredExponential()
    gp = GPRegressor(sexp)
    acq = Acquisition(mode='ExpectedImprovement')
    gpgo = GPGO(gp, acq, wrapper.evaluateLoss, parameter_dict)
    gpgo.run(max_iter=max_iter)

    np.random.seed(seed)
    r = evaluateRandom(gpgo, wrapper, n_eval=max_iter + 1)
    r = cumMax(r)

    np.random.seed(seed)
    sa = evaluateSA(gpgo, wrapper, n_eval=max_iter + 1)
    sa = cumMax(sa)

    return np.array(gpgo.history), r, sa


def plotRes(gpgo_history, random, sa, datasetname):
    import matplotlib.pyplot as plt
    x = np.arange(1, len(random) + 1)
    plt.figure()
    plt.plot(x, -gpgo_history, label='pyGPGO')
    plt.plot(x, -random, label='Random search')
    plt.plot(x, -sa, label='Simulated annealing')
    plt.grid()
    plt.legend(loc=0)
    plt.xlabel('Number of evaluations')
    plt.ylabel('Best log-loss found')
    datasetname = datasetname.split('.')[0]
    plt.title(datasetname)
    plt.savefig(os.path.join(os.path.abspath('.'), 'testing/results/{}.pdf'.format(datasetname)))
    plt.show()
    return None


def evaluateRandom(gpgo, loss, n_eval=20):
    res = []
    for i in range(n_eval):
        param = dict(gpgo._sampleParam())
        l = loss.evaluateLoss(**param)
        res.append(l)
    return (res)


def evaluateSA(gpgo, loss, T=100, cooling=0.9, n_eval=50):
    res = []
    sa = SimulatedAnnealing(gpgo._sampleParam, loss.evaluateLoss, T=T, cooling=cooling)
    sa.run(n_eval)
    return (sa.history)


if __name__ == '__main__':
    #parameter_dict = {'C': ('cont', (0.001, 20)), 'gamma': ('cont', (0.001, 20))}
    #mod = SVM()
    #parameter_dict = {'n_neighbors': ('int', (1, 100)),
    #                  'leaf_size': ('int', (15, 50))}
    #model = KNN()
    parameter_dict = {
    'n_estimators': ('int', (3, 500)),
    'max_features': ('cont', (0.01, 1)),
    'min_samples_split': ('cont', (0.01, 1)),
    'min_samples_leaf': ('cont', (0.01, 0.5))
    }
    model = RF()


    path = os.path.join(os.getcwd(), 'datasets')
    datasets = ['breast_cancer.csv', 'indian_liver.csv', 'parkinsons.csv', 'lsvt.csv', 'pima-indians-diabetes.csv']
    targets = [0, 10, 16, 0, 8]
    for dataset, target in zip(datasets, targets):
        g, r, sa = evaluateDataset(os.path.join(path, dataset), target_index = target, model = model,
                                   parameter_dict = parameter_dict, method = '5fold', seed = 20,
                                   max_iter = 50)
        plotRes(g, r, sa, dataset)
