import os
import tempfile
from pandas.util.testing import assert_series_equal
from cellgrid.ensemble.classifier import *
from cellgrid.model_selection import Evaluator, F1score
from ..conftest import Clf4Test


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


class TestDataMapper:
    def test_create_data_map(self):
        x_train = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]],
                               columns=list('ab'))
        y_train = pd.DataFrame([['n1', 'n11'], ['n2', ''],
                                ['n2', ''], ['n1', 'n12']], columns=['x1', 'x2'])
        dm = DataMapper()
        data_map = dm.create_data_map(x_train, y_train)

        assert_series_equal(data_map['all-events'][1],
                            pd.Series(['n1', 'n2', 'n2', 'n1'], name='x1'))
        assert_series_equal(data_map['n1'][1],
                            pd.Series(['n11', 'n12'], name='x2', index=[0, 3])
                            )


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
