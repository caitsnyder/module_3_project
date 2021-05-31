class ResultsManager:
    def __init__(self, key, results):
        self.clf_name = key.replace("_", " ").upper()
        self.best_estimator = results.best_estimator_
        self.best_score = results.best_score_
        self.best_params = results.best_params_
