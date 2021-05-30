from modules.managers.splits_manager import SplitsManager
from modules.helpers.viz_helper import VizHelper
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from modules.constants.param_grid_constants import param_grids
from modules.constants.classifier_constants import classifiers

class ReportManager:
    def __init__(self, outcome):
        self.splits = None
        self.outcome = outcome
        self.results = {}
        self.models = {}
        
    def run_reports(self, preprocessor, splits: SplitsManager):
        self.splits = splits
        list(map(lambda key: self.execute_pipeline(preprocessor, key), classifiers.keys()))
        list(map(lambda key: self.evaluate_predictions(key), classifiers.keys()))
        # self.display_results()

    def execute_pipeline(self, preprocessor, key):
        param_grid = param_grids[key]
        param_grid['preprocessor__num__imputer__strategy'] = ['mean', 'median']

        pipe = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('clf', classifiers[key])
                        ])
        gs = GridSearchCV(pipe,
                            param_grid,
                            cv=3,
                            # scoring="accuracy",
                            # return_train_score=True,
                            # verbose=1,
                            # n_jobs=-1
                        )
        gs_results  = gs.fit(self.splits.X_train, self.splits.y_train.values.ravel())
        self.results[key] = gs_results.best_score_
        self.models[key] = gs_results.best_estimator_

    def display_results(self):
        print('---------\nAccuracy reports\n---------')
        for key in self.results.keys():
            title = key.replace("_", " ").title()
            print(f"{title}: {self.results[key]:.2%}")

    def evaluate_predictions(self, key):
        print(key)
        y_pred = self.models[key].predict(self.splits.X_test)
        VizHelper().show_confusion_matrix(self.splits.y_test, y_pred, self.outcome, key)


    
            