import os
import tempfile
from pandas.util.testing import assert_series_equal
from cellgrid.ensemble.classifier import *
from cellgrid.model_selection import Evaluator, F1score
from ..conftest import Clf4Test


class TestSchema:
    def test_from_json(self):
        schema = GridSchema.from_json()
        assert isinstance(schema, GridSchema)

    def test_create_model(self):
        bp = ModelBlueprint('test', 'markers', 'xgb', None)
        xgb = GridSchema.create_model(bp)
        assert isinstance(xgb, XgbModel)
        assert xgb._model_ins.n_jobs == 10
        assert xgb._model_ins.max_depth == 10
        assert xgb._model_ins.n_estimators == 40


class TestClassifier:
    def test_fit(self):
        c4t = Clf4Test()
        r = c4t.clf.score(c4t.x_train, c4t.y_train)
        assert list(r.keys()) == ['all-events', 'cells', 'B', 'CD4T',
                                  'CD8T', 'Monocytes', 'NK', 'gdT']
        for v in r.values():
            assert v > 0.9

    def test_predict(self):
        c4t = Clf4Test()
        y = c4t.clf.predict(c4t.x_train)
        for i in range(3):
            label = 'level{}'.format(i)
            assert set(y[label].unique()) == set(c4t.y_train[label].unique())

    def test_save_and_load(self):
        c4t = Clf4Test()
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'model.txt')
            save_model(c4t.clf, f)
            clf2 = load_model(f)
            r = clf2.score(c4t.x_train, c4t.y_train)
            assert list(r.keys()) == ['all-events', 'cells', 'B', 'CD4T',
                                      'CD8T', 'Monocytes', 'NK', 'gdT']
            for v in r.values():
                assert v > 0.9


class TestClfPerformance:
    def test_on_test_dataset(self):
        c4t = Clf4Test()
        r = c4t.clf.score(c4t.x_test, c4t.y_test)
        print(r)

    def test_f1_score(self):
        c4t = Clf4Test()
        f1score = F1score()
        eva = Evaluator(c4t.clf, f1score)
        r = eva(c4t.x_test, c4t.y_test, average='micro')
        print(r)
