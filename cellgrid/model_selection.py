import abc
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize


class Evaluator:
    def __init__(self, clf, method):
        self.clf = clf
        self.method = method

    def __call__(self, x_test, y_test, **kwargs):
        y_pred = self.clf.predict(x_test)
        return self.method(y_test.iloc[:, -1], y_pred.iloc[:, -1],
                           **kwargs)


class EvaMethod(abc.ABC):
    @abc.abstractmethod
    def __call__(self, y_test, y_pred, **kwargs):
        pass


class F1score(EvaMethod):
    def __call__(self, y_test, y_pred, **kwargs):
        return f1_score(y_test, y_pred, **kwargs)


class ConfusionMatrix(EvaMethod):
    def __call__(self, y_test, y_pred, **kwargs):
        return confusion_matrix(y_test, y_pred, **kwargs)


class PrecisionRecallCurve(EvaMethod):
    def __call__(self, y_test, y_pred, **kwargs):
        classes = y_test.unique()
        yt = label_binarize(y_test, classes=classes)
        yp = label_binarize(y_pred, classes=classes)
        return precision_recall_curve(yt.ravel(),
                                      yp.ravel(),
                                      **kwargs)
