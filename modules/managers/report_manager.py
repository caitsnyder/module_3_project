import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.metrics import classification_report


from modules.managers.results_manager import ResultsManager
from modules.managers.splits_manager import SplitsManager
from modules.helpers.viz_helper import VizHelper
from modules.constants.param_grid_constants import param_grids
from modules.constants.classifier_constants import classifiers

class ReportManager:
    def __init__(self, outcome):
        self.splits = None
        self.results = []
        self.outcome = outcome

    def run_reports(self, preprocessor, splits: SplitsManager):
        self.splits = splits
        list(map(lambda key: self.execute_pipeline(preprocessor, key), classifiers.keys()))
        list(map(lambda x: self.display_predictions(x), self.results))
        self.display_results()

    def execute_pipeline(self, preprocessor, key):
        print(f"Beginning {key}...")
        param_grid = param_grids[key]
        param_grid['col__num__imputer__strategy'] = ['mean', 'median']

        pipe = imbPipeline([
            ('col', preprocessor['col']),
            ('sampler', preprocessor['sampler']),
            ("clf", classifiers[key])
            ])
        gs = GridSearchCV(pipe,
                            param_grid,
                            cv=3,
                            scoring="accuracy",
                            n_jobs=-1
                        )
        gs_results  = gs.fit(self.splits.X_train, self.splits.y_train.values.ravel())
        y_preds = gs.predict(self.splits.X_test)
        clf_report = classification_report(self.splits.y_test, y_preds, self.splits.y_train[self.outcome].unique().tolist())
        self.results.append(ResultsManager(key, gs_results, clf_report))

    def display_results(self):
        rows = list(map(lambda x: [x.clf_name, x.best_score, x.best_params], self.results))
        df = pd.DataFrame(rows, columns=[
            'clf_name', 
            'best_score', 
            'best_params'
        ]).sort_values(by='best_score', ascending = False)
        print(df)
        print('---------\nDiagnostics\n---------')
        for result in self.results:
            print(result.clf_name.upper())
            print(f"Accuracy: {result.best_score:.2%}")
            print(f"Classification report:")
            print(result.clf_report)

    def display_predictions(self, result: ResultsManager):
        title = f"{result.clf_name} ({result.best_score:.2%})"
        VizHelper().show_confusion_matrix(result.best_estimator, self.splits.X_test, self.splits.y_test, self.outcome, title)
