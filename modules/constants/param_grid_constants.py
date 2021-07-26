import numpy as np

param_grids = {
    'decision_tree': {
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [None, 2, 3, 4, 5, 6],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 3, 4, 5, 6]
    },
    'random_forest': {
        'clf__n_estimators': [10, 20, 30],
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [5, 10, 20, 30, 35],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [2, 3, 6]
    },
    'knn': {
        'clf__n_neighbors': [3, 5, 7, 11, 19],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan']
    },
    'xgboost': {
        'clf__max_depth': [10],
    }
}