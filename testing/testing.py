from testing.utils import *
from testing.modaux import *

if __name__ == '__main__':

    models = [GBM()]
    params = [d_gbm]


    path = os.path.join(os.getcwd(), 'datasets')
    datasets = ['aff.csv', 'pinter.csv','breast_cancer.csv', 'indian_liver.csv', 'parkinsons.csv',
                'lsvt.csv', 'pima-indians-diabetes.csv']
    problems = ['cont', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary']
    targets = [0, 0, 0, 10, 16, 0, 8]

    for model, parameter_dict in zip(models, params):
        np.random.seed(93)
        print('Evaluating model {}'.format(model.name))
        for dataset, target, problem in zip(datasets, targets, problems):
            np.random.seed(93)
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
