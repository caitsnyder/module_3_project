from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
from modules.managers.splits_manager import SplitsManager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

class VizHelper:
    def __init__(self):
        pass

    def show_visualizations(self, df, outcome):
        cont_features = df.select_dtypes(exclude=['object']).columns
        self.generate_heat_map(df, cont_features)
        self.show_outliers(df, cont_features)
        self.show_basic_correlations(df, cont_features, outcome)

    def generate_heat_map(self, df, features):
        plt.figure(figsize=(7, 6))
        sns.heatmap(df[features].corr(), center=0)
        plt.show()


    def show_outliers(self, df, cols):
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        axe = axes.ravel()

        for i, xcol in enumerate(cols):
            sns.boxplot(x=df[xcol], ax=axe[i])
        plt.show()

    def show_basic_correlations(self, df, cols, outcome):
        preds = [i for i in cols if i != outcome]
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        axe = axes.ravel()

        for i, xcol in enumerate(preds):
            df.plot(kind='scatter', x=xcol, y=outcome, alpha=0.4, color='b', ax=axe[i])
        
        plt.show()

    def show_confusion_matrix(self, clf, X_test, y_test, outcome, title):
        labels = y_test[outcome].unique()
        print(labels)
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                        display_labels=labels,
                                        cmap=plt.cm.Blues)
        disp.ax_.set_title(title)
        print(disp.confusion_matrix)
        plt.show()
