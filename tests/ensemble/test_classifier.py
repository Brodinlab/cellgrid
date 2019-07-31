import os
import pytest
import pandas as pd
from cellgrid.ensemble import *

clf = None


@pytest.fixture(scope="module", autouse=True)
def fit():
    f = os.path.join(os.getcwd(),
                     'tests', 'cellgrid_test.csv')
    df = pd.read_csv(f)
    y_train = df[['level0', 'level1', 'level2']]
    x_train = df.drop(['level0', 'level1', 'level2'], axis=1)
    schema = GridSchema.from_json()
    global clf
    clf = GridClassifier(schema)
    clf.fit(x_train, y_train)


class TestGridSchema:
    def test_from_json(self):
        schema = GridSchema.from_json()
        assert isinstance(schema, GridSchema)
        assert len(schema.get()) == 8
        assert schema.get()[0]['parent'] is None

    def test_fit(self):
        f = os.path.join(os.getcwd(),
                         'tests', 'cellgrid_test.csv')
        df = pd.read_csv(f)
        y_train = df[['level0', 'level1', 'level2']]
        x_train = df.drop(['level0', 'level1', 'level2'], axis=1)
        r = clf.score(x_train, y_train)

        assert list(r.keys()) == ['all-events', 'cells', 'B', 'CD4T',
                                  'CD8T', 'Monocytes', 'NK', 'gdT']
        for v in r.values():
            assert v > 0.9

    def test_predict(self):
        f = os.path.join(os.getcwd(),
                         'tests', 'cellgrid_test.csv')
        df = pd.read_csv(f)
        x_train = df.drop(['level0', 'level1', 'level2'], axis=1)

        y = clf.predict(x_train)
        for i in range(3):
            label = 'level{}'.format(i)
            assert set(y[label].unique()) == set(df[label].unique())

