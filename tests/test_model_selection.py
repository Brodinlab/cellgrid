from cellgrid.model_selection import *
from .conftest import Clf4Test


class TestEvaluator:
    def test_f1(self):
        c4t = Clf4Test()
        n = c4t.y_train.iloc[:, -1].unique().shape[0]
        eva = Evaluator(c4t.clf, F1score())
        r = eva.run(c4t.x_train, c4t.y_train, average=None)
        assert len(r) == n

    def test_confusion_matrix(self):
        c4t = Clf4Test()
        n = c4t.y_train.iloc[:, -1].unique().shape[0]
        eva = Evaluator(c4t.clf, ConfusionMatrix())
        r = eva.run(c4t.x_train, c4t.y_train,
                    labels=c4t.y_train.iloc[:, -1].unique())
        assert r.shape == (n, n)

    def test_precision_recall_curve(self):
        c4t = Clf4Test()
        prc = PrecisionRecallCurve()
        eva = Evaluator(c4t.clf, prc)
        r = eva.run(c4t.x_train, c4t.y_train)
        assert len(r) == 3

