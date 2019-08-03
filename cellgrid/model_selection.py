class Evaluator:
    def __init__(self, clf, method):
        self.clf = clf
        self.method = method

    def run(self, x_test, y_test, **kwargs):
        y_pred = self.clf.predict(x_test)
        return self.method(y_test.iloc[:, -1], y_pred.iloc[:, -1],
                           labels=y_test.iloc[:, -1].unique(), **kwargs)
