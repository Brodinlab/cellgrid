import os
import json
import abc
from collections import namedtuple
from xgboost import XGBClassifier

ModelBlueprint = namedtuple('ModelBlueprint', ['name', 'markers', 'model_class_name', 'parent'])


class Schema(abc.ABC):
    def __init__(self, model_blueprints):
        self._ready = False
        for bp in model_blueprints:
            model = self.create_model(bp)
            self.add_model(model)
        self.build()

    @property
    def ready(self):
        return self._ready

    @abc.abstractmethod
    def create_model(self, model_class_name):
        pass

    @abc.abstractmethod
    def add_model(self, model):
        pass

    @abc.abstractmethod
    def build(self):
        """
        Finish for adding models and set ready flag to true
        :return:
        """

    @abc.abstractmethod
    def create_data_map(self, x_train, y_train):
        """
        Structure the datasets based on the schema
        :return: A dict that maps a model name to the x_train, y_train segments
        """
        pass

    @abc.abstractmethod
    def walk(self):
        """
        Walk through and return each model
        :return:
        """
        pass


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

    def walk(self):
        pass

    def add_model(self, model):
        pass

    def create_data_map(self, x_train, y_train):
        pass

    def create_model(self, model_class_name):
        model_map = {
            'xgb': {
                'model_class': XGBClassifier,
                'params': {
                    'n_jobs': 10,
                    'max_depth': 10,
                    'n_estimators': 40
                }
            }
        }


    def build(self):
        pass
