param_grids = {
    'decision_tree': {
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [None, 2, 3, 4, 5, 6],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 3, 4, 5, 6]
    },
    'random_forest': {
        'clf__n_estimators': [10, 30, 100],
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [None, 2, 6, 10],
        'clf__min_samples_split': [5, 10],
        'clf__min_samples_leaf': [3, 6]
    },
    'svm': {
        'clf__C': [0.1, 1, 10, 100, 1000], 
        'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'clf__kernel': ['rbf']
    },
    'knn': {
        'clf__n_neighbors': [3, 5, 11, 19],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan']
    }


}