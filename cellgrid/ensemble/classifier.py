import os
import pandas as pd
import numpy as np
import pickle
import json
from ..core import AbsDataFrame, AbsSeries, Classifier, Schema, ModelBlueprint
from .model import XgbModel


class DataFrame(AbsDataFrame):
    def __init__(self, data, columns=None, index=None):
        self.df = pd.DataFrame(data,
                               columns=columns,
                               index=index)

    @classmethod
    def from_pd_df(cls, df):
        return DataFrame(df.values,
                         columns=df.columns,
                         index=df.index
                         )

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
            return DataFrame.from_pd_df(self.df.loc[list(index),
                                                    list(columns)])
        elif index is not None:
            return DataFrame.from_pd_df(self.df.loc[list(index)])
        else:
            return DataFrame.from_pd_df(self.df[list(columns)])

    def set_col(self, col_name, series):
        if col_name in self.df:
            self.df.loc[series.index, col_name] = series.values
        else:
            dfs = pd.DataFrame(series.values,
                               index=series.index,
                               columns=[col_name])
            self.df = pd.concat([self.df, dfs], axis=1)

    def drop(self, names, axis):
        df = self.df.drop(names, axis=axis)
        return DataFrame.from_pd_df(df)


class Series(AbsSeries):
    def __init__(self, values, index=None, name=None):
        self.s = pd.Series(values, index=index, name=name)

    @classmethod
    def from_pd_series(cls, s):
        return Series(s.values, index=s.index, name=s.name)

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

    def replace(self, to_replace, value):
        s = self.s.replace(to_replace, value)
        return Series(s.values, index=s.index,
                      name=s.name)

    def drop(self, names):
        return Series.from_pd_series(self.s.drop(names))


class GridClassifier:
    def __init__(self, schema):
        self.clf = Classifier(schema)

    def fit(self, x_train, y_train):
        return self.clf.fit(DataFrame.from_pd_df(x_train),
                            DataFrame.from_pd_df(y_train))

    def score(self, x_test, y_test):
        return self.clf.score(DataFrame.from_pd_df(x_test),
                              DataFrame.from_pd_df(y_test))

    def predict(self, x):
        df = self.clf.predict(DataFrame.from_pd_df(x),
                              DataFrame,
                              Series)
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
