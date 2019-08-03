import numpy as np
import pandas as pd
import pickle


class GridClassifier:
    def __init__(self, schema):
        self.schema = schema

    def fit(self, x_train, y_train):
        dm = DataMapper()
        data_map = dm.create_data_map(x_train, y_train)
        for model in self.schema.walk():
            x, y = data_map[model.name]
            model.fit(x, y)

    def score(self, x_test, y_test):
        r = dict()
        dm = DataMapper()
        data_map = dm.create_data_map(x_test, y_test)
        for model in self.schema.walk():
            x, y = data_map[model.name]
            r[model.name] = model.score(x, y)
        return r

    def predict(self, x):
        df = pd.DataFrame([])
        for model in self.schema.walk():
            parent_level_label = 'level{}'.format(model.level - 1)

            if parent_level_label in df:
                col = df[parent_level_label]
                index = col[col == model.name].index
            else:
                index = x.index

            y = pd.Series(model.predict(x.loc[index]),
                          index=index)

            level_label = 'level{}'.format(model.level)
            if index.shape == x.index.shape:
                df[level_label] = y
            else:
                df.loc[y.index, level_label] = y

        return df.replace(np.nan, ' ')


class DataMapper:
    def create_data_map(self, x_train, y_train):
        r = dict()
        blocks = [
            {'name': 'all-events',
             'index': y_train.index,
             'parent': None}
        ]
        for column in y_train:
            new_blocks = list()
            for block in blocks:
                x_train_block = x_train.loc[block['index']]
                y_train_block = y_train.loc[block['index'], column]
                if len(y_train_block.unique()) != 1:
                    r[block['name']] = x_train_block, y_train_block
                new_blocks += self.__create_new_block(y_train_block,
                                                      block['parent'])
            blocks = new_blocks
        return r

    def __create_new_block(self, y_train, parent):
        series_y = y_train
        blocks = list()
        for name in series_y.unique():
            index = series_y[series_y == name].index
            blocks.append({'name': name,
                           'index': index,
                           'parent': parent})
        return blocks


def save_model(model, path):
    with open(path, 'wb') as fp:
        pickle.dump(model, fp)


def load_model(path):
    with open(path, 'rb') as fp:
        model = pickle.load(fp)
    return model
