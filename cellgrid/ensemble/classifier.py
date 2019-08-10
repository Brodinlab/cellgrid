import os
import pandas as pd
import numpy as np
import pickle
import json
from ..core import AbsDataFrame, AbsSeries, Classifier, Schema, ModelBlueprint
from .model import XgbModel


class DataFrame(AbsDataFrame):
    def __init__(self, df=None):
        if df is None:
            df = pd.DataFrame([])
        self.df = df

    @property
    def index(self):
        return list(self.df.index)

    @property
    def columns(self):
        return list(self.df.columns)

    @property
    def values(self):
        return self.df.values

    def get_col_series(self, name, index=None):
        assert type(name) == str
        if index is None:
            s = self.df[name]
        else:
            s = self.df.loc[list(index), name]
        return Series(s.values, index=s.index, name=s.name)

    def loc(self, index=None, columns=None):
        if index is None and columns is None:
            return DataFrame(self.df)
        elif index is not None and columns is not None:
            return DataFrame(self.df.loc[list(index), list(columns)])
        elif index is not None:
            return DataFrame(self.df.loc[list(index)])
        else:
            return DataFrame(self.df[list(columns)])

    def set_col(self, col_name, series):
        if col_name in self.df:
            self.df.loc[series.index, col_name] = series.values
        else:
            dfs = pd.DataFrame(series.values,
                               index=series.index,
                               columns=[col_name])
            self.df = pd.concat([self.df, dfs], axis=1)


class Series(AbsSeries):
    def __init__(self, values, index=None, name=None):
        self.s = pd.Series(values, index=index, name=name)

    @property
    def index(self):
        return list(self.s.index)

    @property
    def values(self):
        return list(self.s.values)

    def get_item_index(self, item):
        return list(self.s[self.s == item].index)

    def unique(self):
        return list(self.s.unique())


class GridClassifier:
    def __init__(self, schema):
        self.clf = Classifier(schema)

    def fit(self, x_train, y_train):
        return self.clf.fit(DataFrame(x_train), DataFrame(y_train))

    def score(self, x_test, y_test):
        return self.clf.score(DataFrame(x_test), DataFrame(y_test))

    def predict(self, x):
        df = self.clf.predict(DataFrame(x), DataFrame, Series)
        return df.df.replace(np.nan, ' ')


def save_model(model, path):
    with open(path, 'wb') as fp:
        pickle.dump(model, fp)


def load_model(path):
    with open(path, 'rb') as fp:
        model = pickle.load(fp)
    return model


class GridSchema(Schema):
    @classmethod
    def from_json(cls, filepath=None):
        if filepath is None:
            dir_ = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(dir_, 'schema.json')

        with open(filepath, 'rb') as fp:
            items = json.load(fp)

        bps = list()
        for i in items:
            bps.append(ModelBlueprint(**i))
        return GridSchema(bps)

    @classmethod
    def create_model(cls, bp):
        model_map = {
            'xgb': XgbModel
        }
        model_class = model_map[bp.model_class_name]
        return model_class(bp.name, bp.markers, bp.parent)
