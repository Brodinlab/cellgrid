from sklearn.metrics import confusion_matrix, f1_score
from cellgrid.model_selection import Evaluator
from .conftest import Clf4Test


class TestEvaluator:
    def test_run(self):
        c4t = Clf4Test()
        clf, x_train, y_train = [c4t.clf, c4t.x_train, c4t.y_train]
        n = y_train.iloc[:, -1].unique().shape[0]
        eva = Evaluator(clf, f1_score)
        r = eva.run(x_train, y_train, average=None)
        assert len(r) == n

        eva = Evaluator(clf, confusion_matrix)
        r = eva.run(x_train, y_train)
        assert r.shape == (n, n)
