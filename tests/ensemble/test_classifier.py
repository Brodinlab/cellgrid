import os
import tempfile
from cellgrid.ensemble import *
from ..conftest import Clf4Test


class TestClassifier:
    def setup_method(self):
        c4t = Clf4Test()
        self.clf, self.x_train, self.y_train = [c4t.clf, c4t.x_train, c4t.y_train]

    def test_fit(self):
        r = self.clf.score(self.x_train, self.y_train)
        assert list(r.keys()) == ['all-events', 'cells', 'B', 'CD4T',
                                  'CD8T', 'Monocytes', 'NK', 'gdT']
        for v in r.values():
            assert v > 0.9

    def test_predict(self):
        y = self.clf.predict(self.x_train)
        for i in range(3):
            label = 'level{}'.format(i)
            assert set(y[label].unique()) == set(self.y_train[label].unique())

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'model.txt')
            save_model(self.clf, f)
            clf2 = load_model(f)
            r = clf2.score(self.x_train, self.y_train)
            assert list(r.keys()) == ['all-events', 'cells', 'B', 'CD4T',
                                      'CD8T', 'Monocytes', 'NK', 'gdT']
            for v in r.values():
                assert v > 0.9




