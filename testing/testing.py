from testing.utils import *
from testing.modaux import *

if __name__ == '__main__':
    d_rf = {
    'n_estimators': ('int', (3, 500)),
    'max_features': ('cont', (0.01, 1)),
    #'min_samples_split': ('cont', (0.01, 1)),
    'min_samples_leaf': ('cont', (0.01, 0.5))
    }

    d_knn = {
        'n_neighbors': ('int', (1, 100)),
        'leaf_size': ('int', (15, 50))
    }

    d_mlp = {
    'hidden_layer_size': ('int', (5, 200)),
    'alpha': ('cont', (10e-5, 10e-3)),
    'learning_rate_init': ('cont', (10e-5, 10e-3)),
    'beta_1': ('cont', (0.01, 0.99)),
    'beta_2': ('cont', (0.01, 0.99))
    }

    d_svm = {
    'C': ('cont', (0.001, 20)),
    'gamma': ('cont', (0.001, 20))
    }

    models = [RF(), KNN(), MLP(), SVM()]
    params = [d_rf, d_knn, d_mlp, d_svm]

    path = os.path.join(os.getcwd(), 'datasets')
    datasets = ['breast_cancer.csv', 'indian_liver.csv', 'parkinsons.csv', 'lsvt.csv', 'pima-indians-diabetes.csv']
    targets = [0, 10, 16, 0, 8]


    for model, parameter_dict in zip(models, params):
        for dataset, target in zip(datasets, targets):
            g, r, sa = evaluateDataset(os.path.join(path, dataset), target_index = target, model = model,
                                       parameter_dict = parameter_dict, method = '5fold', seed = 20,
                                       max_iter = 50)
            plotRes(g, r, sa, dataset, model)