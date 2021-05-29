from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from modules.constants.param_grid_constants import param_grids
from modules.constants.classifier_constants import classifiers

class ReportManager:
    def __init__(self):
        self.splits = None
        self.results = {}
        
    def run_reports(self, preprocessor, splits):
        self.splits = splits
        list(map(lambda key: self.execute_pipeline(preprocessor, key), classifiers.keys()))
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
        
    def display_results(self):
        print('---------\nAccuracy reports\n---------')
        for key in self.results.keys():
            title = key.replace("_", " ").title()
            print(f"{title}: {self.results[key]:.2%}")
