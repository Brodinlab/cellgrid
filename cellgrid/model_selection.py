from sklearn.metrics import confusion_matrix, f1_score
from .ensemble.classifier import DataFrame, Series
from .core import EvaMethod


class Evaluator:
    def __init__(self, clf, method):
        self.clf = clf
        self.method = method

    def __call__(self, x_test, y_test, **kwargs):
        y_pred = self.clf.predict(x_test)
        return self.method.run(DataFrame.from_pd_df(y_test),
                               DataFrame.from_pd_df(y_pred),
                               **kwargs)


class F1score(EvaMethod):
    def __init__(self):
        super(F1score, self).__init__(DataFrame,
                                      Series)

    @property
    def meta_method(self):
        return f1_score


class ConfusionMatrix(EvaMethod):
    def __init__(self):
        super(ConfusionMatrix, self).__init__(DataFrame,
                                              Series)

    @property
    def meta_method(self):
        return confusion_matrix
