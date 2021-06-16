from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class VizHelper:
    def __init__(self):
        pass
   
    def show_visualizations(self, df, outcome):
        cont_features = df.select_dtypes(exclude=['object']).columns
        self.check_outcome_distribution(df, outcome)
        self.generate_heat_map(df, cont_features)
        self.show_outliers(df, cont_features)
        self.show_basic_correlations(df, cont_features, outcome)
        self.show_outcome_dist(df, outcome)

    def check_outcome_distribution(self, df, outcome):
        labels = df[outcome].value_counts().index
        cnts = df[outcome].value_counts().values
        
        df_temp = pd.DataFrame({'labels':labels, 'counts':cnts})
        ax = df_temp.plot.bar(x='labels', y='counts', rot=0)
        ax.set_title("Frequency of outcome values")


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

    def show_outcome_dist(self, df, outcome):
        df[outcome].value_counts().plot(kind='bar')

    def show_confusion_matrix(self, clf, X_test, y_test, outcome, title):
        labels = y_test[outcome].unique()
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                        display_labels=labels,
                                        cmap=plt.cm.Greens,
                                        xticks_rotation='vertical')
        disp.ax_.set_title(title)
        plt.show()
