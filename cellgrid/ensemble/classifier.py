import os
import json
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


class Schema:
    def __init__(self, schema):
        self.schema = schema

    def get(self):
        return self.schema


class GridSchema(Schema):
    @classmethod
    def from_json(cls, filepath=None):
        if filepath is None:
            filepath = os.path.join(os.getcwd(), 'cellgrid',
                                    'ensemble', 'schema.json')

        with open(filepath, 'rb') as fp:
            schema = json.load(fp)
        return GridSchema(schema)
