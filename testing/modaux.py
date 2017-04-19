from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from collections import OrderedDict

d_rf = OrderedDict()
d_rf['n_estimators'] = ('int', (10, 50))
d_rf['min_samples_split'] = ('cont', (0.1, 0.5))
d_rf['max_features'] = ('cont', (0.1, 0.5))

d_knn = OrderedDict()
d_knn['n_neighbors'] = ('int', (10, 50))

d_mlp = OrderedDict()
d_mlp['hidden_layer_size'] = ('int', (5, 50))
d_mlp['alpha'] = ('cont', (1e-5, 0.9))


d_svm = OrderedDict()
d_svm['C'] = ('cont', (-4, 5))
d_svm['gamma'] = ('cont', (-4, 5))

d_tree = OrderedDict()
d_tree['max_features'] = ('cont', (0.1, 0.99))
d_tree['max_depth'] = ('int', (4, 30))
d_tree['min_samples_split'] = ('cont', (0.1, 0.99))

d_ada = OrderedDict()
d_ada['n_estimators'] = ('int', (5, 200))
d_ada['learning_rate'] = ('cont', (1e-5, 1))

d_gbm = OrderedDict()
d_gbm['learning_rate'] = ('cont', (10e-5, 1e-1))
d_gbm['n_estimators'] = ('int', (10, 100))
d_gbm['max_depth'] = ('int', (2, 100))
d_gbm['min_samples_split'] = ('int', (2, 100))



class Tree:
    def __init__(self, problem='binary', max_features=0.5, max_depth=1, min_samples_split=2):
        self.problem = problem
        self.max_features = max_features
        self.max_depth = int(max_depth)
        self.min_samples_split = min_samples_split
        self.name = 'Tree'

    def eval(self):
        if self.problem == 'binary':
            mod = DecisionTreeClassifier(max_features=self.max_features, max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split, random_state=20)
        else:
            mod = DecisionTreeRegressor(max_features=self.max_features, max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split, random_state=20)
        return mod


class Ada:
    def __init__(self, problem='binary', n_estimators=50, learning_rate=1):
        self.problem = problem
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.name = 'Ada'

    def eval(self):
        if self.problem == 'binary':
            mod = AdaBoostClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                     random_state=20)
        else:
            mod = AdaBoostRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                    random_state=20)
        return mod


class GBM:
    def __init__(self, problem='binary', learning_rate=0.1, n_estimators=100, max_depth=3, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, subsample=1.0, max_features=1.0):
        self.problem = problem
        self.learning_rate = learning_rate
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.name = 'GBM'

    def eval(self):
        if self.problem == 'binary':
            mod = GradientBoostingClassifier(learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                                             max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf,
                                             min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                             subsample=self.subsample,
                                             max_features=self.max_features,
                                             random_state=20)
        else:
            mod = GradientBoostingRegressor(learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                                            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                            min_samples_leaf=self.min_samples_leaf,
                                            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                            subsample=self.subsample,
                                            max_features=self.max_features,
                                            random_state=20)
        return mod


class SVM:
    def __init__(self, problem='binary', C=0, gamma=0, kernel='rbf'):
        self.problem = problem
        self.C = 10**C
        self.gamma = 10**gamma
        self.kernel = kernel
        self.name = 'SVM'

    def eval(self):
        if self.problem == 'binary':
            mod = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=True, random_state=20)
        else:
            mod = SVR(kernel=self.kernel, C=self.C, gamma=self.gamma)
        return mod


class RF:
    def __init__(self, problem='binary', n_estimators=10, max_features=0.5,
                 min_samples_split=0.3, min_samples_leaf=0.2):
        self.problem = problem
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.name = 'RF'

    def eval(self):
        if self.problem == 'binary':
            mod = RandomForestClassifier(n_estimators=self.n_estimators,
                                         max_features=self.max_features,
                                         min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,
                                         n_jobs=-1,
                                         random_state=20)
        else:
            mod = RandomForestRegressor(n_estimators=self.n_estimators,
                                        max_features=self.max_features,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        n_jobs=-1,
                                        random_state=20)
        return mod


class KNN:
    def __init__(self, problem='binary', n_neighbors=5, leaf_size=30):
        self.problem = problem
        self.n_neighbors = int(n_neighbors)
        self.leaf_size = int(leaf_size)
        self.name = 'KNN'

    def eval(self):
        if self.problem == 'binary':
            mod = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                       leaf_size=self.leaf_size)
        else:
            mod = KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                      leaf_size=self.leaf_size)
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
            mod = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                learning_rate_init=self.learning_rate_init, beta_1=self.beta_1, beta_2=self.beta_2,
                                random_state=20)
        else:
            mod = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                               learning_rate_init=self.learning_rate_init, beta_1=self.beta_1, beta_2=self.beta_2,
                               random_state=20)
        return mod
