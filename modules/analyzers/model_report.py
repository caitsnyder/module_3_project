from modules.analyzers.splitter import Splitter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from scipy import stats
# from sklearn import tree
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from modules.analyzers.param_grid_constants import param_grids
from modules.analyzers.classifier_constants import classifiers

class ModelReport:
    def __init__(self):
        self.splits = None
        self.results = {}
        
    def run_reports(self, preprocessor, splits: Splitter, ):
        self.splits = splits
        self.execute_pipeline(preprocessor, 'decision_tree')
        # self.report_decision_tree(pipe, 'decision_tree')
        # self.report_random_forest(pipe, 'random_forest')
        # self.report_svm(pipe, 'svm')
        # self.report_knn(pipe, 'knn')
        self.display_results()

    def execute_pipeline(self, preprocessor, key):

        clf = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('classifier', classifiers[key])
                        ])

        gs = GridSearchCV(clf,
                            param_grids[key],
                            cv=3,
                            scoring="accuracy",
                            return_train_score=True,
                            verbose=1,
                            n_jobs=-1
                        )
        gs_results  = gs.fit(self.splits.X_train, self.splits.y_train.values.ravel())
        self.results[key] = gs_results.best_score_
        
    # def report_decision_tree(self, pipe):
    #     gs = GridSearchCV(DecisionTreeClassifier(), param_grids['decision_tree'], cv=3, scoring="accuracy", return_train_score=True)
    #     self.set_best_score(gs, 'decision_tree')

    # def report_random_forest(self, pipe):
    #     gs = GridSearchCV(RandomForestClassifier(), param_grids['random_forest'], cv=3, scoring="accuracy")
    #     self.set_best_score(gs, 'random_forest')
    #     # model = rf_grid_search.best_estimator_
    #     # confus matrix

    # def report_svm(self, pipe):
    #     gs = GridSearchCV(SVC(), param_grids['svm'], refit = True, verbose = 1)
    #     self.set_best_score(gs, 'svm')
        
    # def report_knn(self, pipe):
    #     gs = GridSearchCV(KNeighborsClassifier(), param_grids['knn'], verbose=1, cv=3, n_jobs=-1)
    #     self.set_best_score(gs, 'knn')

    def display_results(self):
        print('---------\nAccuracy reports\n---------')
        for key in self.results.keys():
            title = key.replace("_", " ").title()
            print(f"{title}: {self.results[key]:.2%}")




    
    # def get_features(self):
    #     self.features = self.splits.X_train.columns.values.tolist() + \
    #         self.splits.y_train.columns.values.tolist()