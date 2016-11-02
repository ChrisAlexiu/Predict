
"""
PredictQuali: (predict qualitative)
1. produce a collection of models (multiple model types, each with multiple combinations of parameter sets) for predicting a binary qualitative/categorical/discrete outcome,
2. evaluate each of them with several metrics,
3. and then dump all results to a CSV file.

PredictQuali uses pandas and scikit-learn.

Usage:
    - import predictquali
    - object = PredictQuali()
    - # object.show_info()
    - object.results(X, y)
    - object.to_csv(filepath) # via pandas

Arguments:
    - PredictQuali can be instantiated without arguments - defaults will be used
    - use none or one of 'grid_keep' or 'grid_drop' - do not use both
    - when using 'grid_keep' or 'grid_drop':
        - input model type name short form as a list
        - lower case is acceptable

Output: use 'object.results()' to do the modeling and return results as a pandas DataFrame. Follow up with '.to_csv()' to save results to a CSV file.
"""

import itertools
import collections
import copy
import time
import pandas
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import neural_network

class PredictQuali(object):
    
    class_title_error = "ERROR re: class PredictQuali\n • "

    def __init__(self, grid=None, grid_keep=None, grid_drop=None, scor=None, CV_k=5, \
            n_processes=1, verbose=True):
        self.setup_grid_scor(grid, grid_keep, grid_drop, scor)
        self.kFoldCV = model_selection.KFold(n_splits=CV_k, shuffle=False)
        self.n_processes = n_processes
        self.verbose = verbose
    
    def setup_grid_scor(self, grid=None, grid_keep=None, grid_drop=None, scor=None):
    # setter method for grid (+ grid_keep, grid_drop) and score - also, some checks
        self.grid = self.default_grid() if grid == None else grid
        self.scor = self.default_scor() if scor == None else scor
        # convert 'grid_keep' and 'grid_drop' to uppercase
        if grid_keep != None: grid_keep = [x.upper() for x in grid_keep]
        if grid_drop != None: grid_drop = [x.upper() for x in grid_drop]
        # checks:
        if grid_keep != None and grid_drop != None:
            quit(PredictQuali.class_title_error + \
            "Use only one of 'grid_keep' or 'grid_drop'.")
        if grid_keep != None and \
            set(grid_keep).issubset(set([x[0] for x in self.grid])) == False:
            quit(PredictQuali.class_title_error + \
            "Invalid input for parameter 'grid_keep' with respect to 'grid'.")
        if grid_drop != None and \
            set(grid_drop).issubset(set([x[0] for x in self.grid])) == False:
            quit(PredictQuali.class_title_error + \
            "Invalid input for parameter 'grid_drop' with respect to 'grid'.")
        # filter grid using 'grid_keep' or 'grid_drop':
        if grid_keep != None:
            self.grid = [x for x in self.grid if x[0] in grid_keep]
        if grid_drop != None:
            self.grid = [x for x in self.grid if x[0] not in grid_drop]
        return None
    
    def setup_pars(self, grid):
    # get cartesian product for parameters
        for x in grid:
            i = 3
            x[i] = [zip(itertools.repeat(k,len(v)),v) for k,v in x[i].items()]
            x[i] = [collections.OrderedDict(y) for y in itertools.product(*x[i])]
            x.append(len(x[i]))
        return grid
    
    def setup_head(self):
    # get list of headings for results set
        results_headings = "Model_Type Parameters Duration_(s)".split()
        results_headings.extend(self.scor.keys())
        return results_headings

    def show_info(self):
        temp_grid = self.setup_pars(copy.deepcopy(self.grid))
        print("="*80)
        print("Binary Classification Modeling")
        print("-"*80)
        print("Model Types:", len(temp_grid))
        print("Models:", sum([x[-1] for x in temp_grid]), "•",end=" ")
        print(", ".join( \
            [(n_sh+":"+str(qnty)) for n_sh,n_lo,etmr,pars,qnty in temp_grid]))
        print("Metrics:", ", ".join(self.scor.keys()))
        print("="*80)
        return None

    def results(self, X, y):
    # self.level1_control() returns a pandas dataframe -> save it to CSV format
        self.data_x, self.data_y = X, y
        self.grid = self.setup_pars(self.grid)
        # self.level1_control().to_csv(saveto)
        # return None
        return self.level1_control()

    def level1_control(self):
    # flatten grid and iterate through all model x parameters sets
        import multiprocessing
        results = []
        iterthis = [(itertools.repeat(n_lo,qnty), itertools.repeat(etmr,qnty), \
            pars) for n_sh,n_lo,etmr,pars,qnty in self.grid]
        iterthis = [zip(*x) for x in iterthis]
        iterthis = itertools.chain.from_iterable(iterthis)
        mp_pool = multiprocessing.Pool(processes=self.n_processes)
        results.extend(mp_pool.map(self.level2_modeling, iterthis))
        mp_pool.close()
        mp_pool.join()
        results = pandas.DataFrame(dict(zip(self.setup_head(), zip(*results))))
        results = results[self.setup_head()]
        results.index = range(1, results.shape[0]+1)
        results.index.name = "id"
        return results

    def level2_modeling(self, iterthis):
    # do modeling here -> transfered here via multiprocess pool map from level1
        time_start = time.time()
        n_lo, etmr, pars = iterthis
        if self.verbose == True: print(n_lo, pars)
        scores = collections.OrderedDict([(k,[]) for k,v in self.scor.items()])
        for train, test in self.kFoldCV.split(X=self.data_x):
            preds = etmr(**pars) \
                .fit(X=self.data_x.ix[train], y=self.data_y.ix[train]) \
                .predict(X=self.data_x.ix[test])
            cm = metrics.confusion_matrix(self.data_y.ix[test], preds)
            tpn, tnn, fpn, fnn = cm[1][1], cm[0][0], cm[0][1], cm[1][0]
            tpp, tnp = tpn/(tpn+fpn), tnn/(tnn+fpn)
            fpp, fnp = fpn/(fpn+tnn), fnn/(fnn+tpn)
            scr = dict(zip([k for k,v in self.scor.items() if v==None], \
                [tpn, tnn, fpn, fnn, tpp, tnp, fpp, fnp]))
            for k,v in self.scor.items():
                if v==None: scores[k].append(scr[k])
                if v!=None: scores[k].append(v(self.data_y.ix[test], preds))
        scores = collections.OrderedDict([(k,sum(v)/len(v)) for k,v in scores.items()])       
        return ( n_lo, pars, time.time()-time_start, *scores.values() )
    
    def default_grid(self):
    # provide default for 'grid'
        # grid =
            # name_short, name_long
                # estimator
                # parameters
                # (quantity)
        default_grid = [
            ['LR', 'Logistic_Regression',
                linear_model.LogisticRegression,
                {'penalty':['l1', 'l2'],
                'C':[1, 10, 100, 200],
                }
            ],
            ['SV', 'Support_Vector_Machine',
                svm.SVC,
                {'C':[1, 10, 20],
                'kernel':['linear', 'rbf', 'sigmoid', 'poly'],
                }
            ],
            ['GD', 'Stochastic_Gradient_Descent',
                linear_model.SGDClassifier,
                {'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                'penalty':['l1', 'l2', 'elasticnet'],
                'n_iter':[5, 10, 15, 20],
                }
            ],
            ['AB', 'AdaBoost',
                ensemble.AdaBoostClassifier,
                {'n_estimators':[10, 20, 50],
                'algorithm':['SAMME', 'SAMME.R'],
                }
            ],
            ['ET', 'Extra_Trees',
                ensemble.ExtraTreesClassifier,
                {'n_estimators':[10, 20, 50],
                'criterion':['gini', 'entropy'],
                'max_features':[None, 'auto', 'sqrt', 'log2'],
                'min_samples_split':[1, 2, 3],
                }
            ],
            ['GB', 'Gradient_Boost',
                ensemble.GradientBoostingClassifier,
                {'loss':['deviance', 'exponential'],
                # 'learning_rate':[0.1, 1, 10],
                'n_estimators':[100, 150, 200],
                'max_depth':[2, 3, 4],
                # 'criterion':['friedman_mse', 'mse', 'mae'],
                'max_features':[None, 'auto', 'sqrt', 'log2'],
                }
            ],
            ['RF', 'Random_Forest',
                ensemble.RandomForestClassifier,
                {'n_estimators':[10, 20, 50, 100],
                'criterion':['gini', 'entropy'],
                'max_features':[None, 'auto', 'sqrt', 'log2'],
                'min_samples_split':[1, 2, 3],
                }
            ],
            ['NB', 'Naive_Bayes',
                naive_bayes.GaussianNB,
                {}
            ],
            ['KN', 'kNN',
                neighbors.KNeighborsClassifier,
                {'n_neighbors':[3, 5, 7, 9, 11],
                'weights':['uniform', 'distance'],
                'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
               # 'leaf_size':[20, 30, 40],
               'p':[1, 2, 3],
                }
            ],
            ['NN', 'Neural_Network',
                neural_network.MLPClassifier,
                {'activation':['identity', 'logistic', 'tanh', 'relu'],
                'solver':['lbfgs', 'sgd', 'adam'],
                }
            ],
        ]        
        return default_grid
    
    def default_scor(self):
    # provide default for 'scoring' (metrics for evaluating models)
        default_scor = collections.OrderedDict([
            ("True_Pos_n", None),
            ("True_Neg_n", None),
            ("Flse_Pos_n", None),
            ("Flse_Neg_n", None),
            ("True_Pos_p", None),
            ("True_Neg_p", None),
            ("Flse_Pos_p", None),
            ("Flse_Neg_p", None),
            ("Precision",     metrics.precision_score),
            ("Recall",        metrics.recall_score),
            ("F1_Score",      metrics.f1_score),
            ("Accuracy",      metrics.accuracy_score),
            ("Cohens_Kappa",  metrics.cohen_kappa_score),
            ("Matthews_Corr", metrics.matthews_corrcoef),
        ])
        return default_scor