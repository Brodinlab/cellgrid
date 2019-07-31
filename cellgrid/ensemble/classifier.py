import os
import json
import pickle
from .node import NodesTrainer


class GridClassifier:
    def __init__(self, schema):
        self.nt = NodesTrainer(schema)
        self.nt.schema_to_nodes()

    def fit(self, x_train, y_train):
        self.nt.update_node_data(x_train, y_train)
        self.nt.fit()

    def score(self, x_test, y_test):
        self.nt.update_node_data(x_test, y_test)
        return self.nt.score()

    def predict(self, x):
        return self.nt.predict(x)


class Schema:
    def __init__(self, schema):
        self.__schema = schema

    def get(self):
        return self.__schema

    @classmethod
    def from_json(cls, filepath=None):
        if filepath is None:
            filepath = os.path.join(os.getcwd(), 'cellgrid',
                                    'ensemble', 'schema.json')

        with open(filepath, 'rb') as fp:
            schema = json.load(fp)
        return Schema(schema)


def save_model(model, path):
    with open(path, 'wb') as fp:
        pickle.dump(model, fp)


def load_model(path):
    with open(path, 'rb') as fp:
        model = pickle.load(fp)
    return model
