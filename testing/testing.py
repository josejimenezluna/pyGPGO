import numpy as np
from testing.utils import *
from testing.modaux import *

if __name__ == '__main__':
    d_rf = {
    'n_estimators': ('int', (10, 200)),
    'max_depth': ('int', (2, 20)),
    'min_samples_split': ('cont', (0.01, 1)),
    #'min_samples_leaf': ('cont', (0.01, 0.5)),
    'subsample': ('cont', (0.01, 0.99)),
    'max_features': ('cont', (0.01, 1))
    }

    d_knn = {
        'n_neighbors': ('int', (1, 8)),
        'leaf_size': ('int', (15, 50))
    }

    d_mlp = {
    'hidden_layer_size': ('int', (5, 200)),
    'alpha': ('cont', (10e-5, 10e-2)),
    'learning_rate_init': ('cont', (10e-5, 10e-2)),
    'beta_1': ('cont', (0.001, 0.999)),
    'beta_2': ('cont', (0.001, 0.999))
    }

    d_svm = {
    'C': ('cont', (0.01, 200)),
    'gamma': ('cont', (0.001, 10))
    }

    d_tree = {
        'max_features': ('cont', (0.1, 0.99)),
        'max_depth': ('int', (4, 30)),
        'min_samples_split': ('cont', (0.1, 0.99))
    }

    d_ada = {
        'n_estimators': ('int', (5, 200)),
        'learning_rate': ('cont', (0.01, 10))
    }

    d_gbm = {
    'learning_rate': ('cont', (10e-5, 1)),
    'n_estimators': ('int', (10, 300)),
    'max_depth': ('int', (2, 25)),
    'min_samples_split': ('int', (2, 25)),
    'min_samples_leaf': ('int', (2, 25)),
    #'min_weight_fraction_leaf': ('cont', (0.01, 0.49)),
    'subsample': ('cont', (0.01, 0.99)),
    'max_features': ('cont', (0.01, 0.99))
    }

    models = [GBM()]
    params = [d_gbm]


    path = os.path.join(os.getcwd(), 'datasets')
    datasets = ['aff.csv','breast_cancer.csv', 'indian_liver.csv', 'parkinsons.csv',
                'lsvt.csv', 'pima-indians-diabetes.csv']
    problems = ['cont', 'binary', 'binary', 'binary', 'binary', 'binary']
    targets = [0, 0, 10, 16, 0, 8]

    np.random.seed(20)
    for model, parameter_dict in zip(models, params):
        print('Evaluating model {}'.format(model.name))
        for dataset, target, problem in zip(datasets, targets, problems):
            if problem == 'cont':
                model = model.__class__(problem='cont')
            else:
                model = model.__class__(problem='binary')
            try:
                g, g2, g3, r= evaluateDataset(os.path.join(path, dataset), target_index = target, model = model,
                                           parameter_dict = parameter_dict, method = '5fold', seed = 20,
                                           max_iter = 50, problem=problem)
                plotRes(g, g2, g3, r, dataset, model, problem=problem)
            except Exception as e:
                print(e)
                continue
