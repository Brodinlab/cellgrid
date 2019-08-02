import os
import pytest
import tempfile
import pandas as pd
from cellgrid.ensemble import *

clf = None
x_train = None
y_train = None


@pytest.fixture(scope="module", autouse=True)
def fit():
    dir_ = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(dir_, 'cellgrid_test.csv')
    df = pd.read_csv(f)
    global clf, x_train, y_train

    y_train = df[['level0', 'level1', 'level2']]
    x_train = df.drop(['level0', 'level1', 'level2'], axis=1)
    schema = GridSchema.from_json()
    clf = GridClassifier(schema)
    clf.fit(x_train, y_train)


class TestClassifier:
    def test_fit(self):
        r = clf.score(x_train, y_train)
        assert list(r.keys()) == ['all-events', 'cells', 'B', 'CD4T',
                                  'CD8T', 'Monocytes', 'NK', 'gdT']
        for v in r.values():
            assert v > 0.9

    def test_predict(self):
        y = clf.predict(x_train)
        for i in range(3):
            label = 'level{}'.format(i)
            assert set(y[label].unique()) == set(y_train[label].unique())

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'model.txt')
            save_model(clf, f)
            clf2 = load_model(f)
            r = clf2.score(x_train, y_train)
            assert list(r.keys()) == ['all-events', 'cells', 'B', 'CD4T',
                                      'CD8T', 'Monocytes', 'NK', 'gdT']
            for v in r.values():
                assert v > 0.9




