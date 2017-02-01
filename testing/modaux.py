from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

"""
Parameter suggestion __
SVM:
    D_svm = {
    'C': ('cont', (0.001, 20))
    'gamma': ('cont', (0.001, 20))
    }

RF:
    D_rf = {
    'n_estimators': ('int', (3, 500)),
    'max_features': ('cont', (0.01, 1)),
    'min_samples_split': ('cont', (0.01, 1)),
    'min_samples_leaf': ('cont', (0.01, 1))
    }

KNN:
    D_knn = {
    'n_neighbors': ('int', (1, 100)),
    'leaf_size': ('int', (15, 50))
}

MLP:
    D_mlp = {
    'hidden_layer_size': ('int', (5, 200)),
    'alpha': ('cont', (10e-5, 10e-3)),
    'learning_rate_init': ('cont', (10e-5, 10e-3)),
    'beta_1': ('cont', (0.01, 0.99)),
    'beta_2': ('cont', (0.01, 0.99))
    }
"""


class SVM:
    def __init__(self, problem = 'binary', C = 1.0, gamma = 1.0):
        self.problem = problem
        self.C = C
        self.gamma = gamma
        self.name = 'SVM'
    def eval(self):
        if self.problem == 'binary':
            mod = SVC(C = self.C, gamma = self.gamma, probability = True)
        else:
            mod = SVR(C = self.C, gamma = self.gamma)
        return mod


class RF:
    def __init__(self, problem = 'binary', n_estimators = 10, max_features = 0.5,
                 min_samples_split = 0.3, min_samples_leaf = 0.2):
        self.problem = problem
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.name = 'RF'
    def eval(self):
        if self.problem == 'binary':
            mod = RandomForestClassifier(n_estimators = self.n_estimators,
                                         max_features = self.max_features,
                                         min_samples_split = self.min_samples_split,
                                         min_samples_leaf = self.min_samples_leaf,
                                         n_jobs = -1)
        else:
            mod = RandomForestRegressor(n_estimators = self.n_estimators,
                                        max_features = self.max_features,
                                        min_samples_split = self.min_samples_split,
                                        min_samples_leaf = self.min_samples_leaf,
                                        n_jobs = -1)
        return mod


class KNN:
    def __init__(self, problem = 'binary', n_neighbors = 5, leaf_size = 30):
        self.problem = problem
        self.n_neighbors = int(n_neighbors)
        self.leaf_size = int(leaf_size)
        self.name = 'KNN'
    def eval(self):
        if self.problem == 'binary':
            mod = KNeighborsClassifier(n_neighbors = self.n_neighbors,
                                       leaf_size = self.leaf_size)
        else:
            mod = KNeighborsRegressor(n_neighbors = self.n_neighbors,
                                      leaf_size = self.leaf_size)
        return mod

class MLP:
    def __init__(self, problem='binary', hidden_layer_size=100, alpha=10e-4,
                learning_rate_init=10e-4, beta_1=0.9, beta_2=0.999):
        self.problem = problem
        self.hidden_layer_sizes = (int(hidden_layer_size),)
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.name = 'MLP'
    def eval(self):
        if self.problem == 'binary':
            mod = MLPClassifier(hidden_layer_sizes = self.hidden_layer_sizes, alpha = self.alpha,
                                learning_rate_init = self.learning_rate_init, beta_1 = self.beta_1, beta_2 = self.beta_2)
        else:
            mod = MLPRegressor(hidden_layer_sizes = self.hidden_layer_sizes, alpha = self.alpha,
                                learning_rate_init = self.learning_rate_init, beta_1 = self.beta_1, beta_2 = self.beta_2)
        return mod
