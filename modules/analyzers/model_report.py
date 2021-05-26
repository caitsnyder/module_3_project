import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import tree
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class ModelReport:
    def __init__(self):
        self.splits = None
        self.accuracy_report = {'tree': 0, 'svm': 0, 'knn': 0}
    
    def get_reports(self, splits):
        self.splits = splits
        self.get_decision_tree_report()
        self.get_svm_report()
        self.get_knn_report()
        self.display_results()
        
    def get_decision_tree_report(self):
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(self.splits['X_train'], self.splits['y_train'])
        y_pred_tree = clf.predict(self.splits['X_test'])
        self.accuracy_report['tree'] = accuracy_score(self.splits['y_test'], y_pred_tree)

    def show_decision_tree_plot(self, outcome_values, clf): # Not actively used
        features = self.get_features()
        fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (3,3), dpi=300)
        tree.plot_tree(clf,
                       feature_names=features, 
                       class_names=outcome_values.astype('str'),
                       filled = True)
        plt.show()
        
    def get_features(self):
        self.features = self.splits['X_train'].columns.values.tolist() + \
            self.splits['y_train'].columns.values.tolist()
        
    def get_svm_report(self):
        SVC_model = SVC()
        SVC_model.fit(self.splits['X_train'], self.splits['y_train'].values.ravel())
        SVC_prediction = SVC_model.predict(self.splits['X_test'])
        self.accuracy_report['svm'] = accuracy_score(SVC_prediction, self.splits['y_test'])
    
    def get_knn_report(self):
        KNN_model = KNeighborsClassifier(n_neighbors=5)
        KNN_model.fit(self.splits['X_train'], self.splits['y_train'].values.ravel())
        KNN_prediction = KNN_model.predict(self.splits['X_test'])
#         print(confusion_matrix(SVC_prediction, dfs.splits['y_test']))
#         print(classification_report(KNN_prediction, dfs.splits['y_test']))
        self.accuracy_report['knn'] = accuracy_score(KNN_prediction, self.splits['y_test'])

    def display_results(self):
        print('---------\nAccuracy reports\n---------')
        for key in self.accuracy_report.keys():
            print(key, self.accuracy_report[key])
    