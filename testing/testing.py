from testing.utils import evaluateDataset, plotRes
from testing.modaux import *
import random
import os
import numpy as np

if __name__ == '__main__':
    random.seed(93)
    models = [MLP()]
    params = [d_mlp]

    path = os.path.join(os.getcwd(), 'datasets')
    #datasets = ['aff.csv', 'pinter.csv', 'breast_cancer.csv', 'indian_liver.csv', 'parkinsons.csv',
    #            'lsvt.csv', 'pima-indians-diabetes.csv']
    #problems = ['cont', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary']
    #targets = [0, 0, 0, 10, 16, 0, 8]
    datasets = ['aff.csv']
    problems = ['cont']
    targets = [0]

    for model, parameter_dict in zip(models, params):
        print('Evaluating model {}'.format(model.name))
        for dataset, target, problem in zip(datasets, targets, problems):
            model.problem = problem
            np.random.seed(93)
            print(np.random.randn(1))
            try:
                g, g2, g3, r = evaluateDataset(os.path.join(path, dataset), target_index=target, model=model,
                                               parameter_dict=parameter_dict, method='5fold', seed=20,
                                               max_iter=50, problem=problem)

                plotRes(g, g2, g3, r, dataset, model, problem=problem)
                print(np.random.randn(1))
            except Exception as e:
                print(e)
                continue
