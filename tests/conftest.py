import os
import pytest
import pandas as pd
from cellgrid.ensemble import *


class Clf4Test:
    class __OnlyOne:
        def __init__(self, clf=None, x_train=None, y_train=None):
            self._clf = clf
            self._x_train = x_train
            self._y_train = y_train

        @property
        def clf(self):
            return self._clf

        @property
        def x_train(self):
            return self._x_train

        @property
        def y_train(self):
            return self._y_train

    instance = None

    def __init__(self, **kwargs):
        if not Clf4Test.instance:
            Clf4Test.instance = Clf4Test.__OnlyOne(**kwargs)

    def __getattr__(self, name):
        return getattr(self.instance, name)


@pytest.fixture(scope="session", autouse=True)
def fit():
    dir_ = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(dir_, 'ensemble', 'cellgrid_test.csv')
    df = pd.read_csv(f)
    y_train = df[['level0', 'level1', 'level2']]
    x_train = df.drop(['level0', 'level1', 'level2'], axis=1)
    schema = GridSchema.from_json()
    clf = GridClassifier(schema)
    clf.fit(x_train, y_train)
    Clf4Test(clf=clf, x_train=x_train, y_train=y_train)
