from cellgrid.model_selection import *
from .conftest import Clf4Test


class TestEvaluator:
    def test_f1(self):
        c4t = Clf4Test()
        eva = Evaluator(c4t.clf, F1score())
        r = eva(c4t.x_test, c4t.y_test, average=None)
        assert len(r) == 8

    def test_confusion_matrix(self):
        c4t = Clf4Test()
        eva = Evaluator(c4t.clf, ConfusionMatrix())
        r = eva(c4t.x_test, c4t.y_test)
        assert len(r) == 8
