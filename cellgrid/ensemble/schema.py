import os
import json
import abc
from collections import namedtuple
from .model import *

ModelBlueprint = namedtuple('ModelBlueprint', ['name', 'markers', 'model_class_name', 'parent'])


class Schema(abc.ABC):
    def __init__(self, model_blueprints):
        self._ready = False
        for bp in model_blueprints:
            model = Schema.create_model(bp)
            self.add_model(model)
        self.build()

    @property
    def ready(self):
        return self._ready

    @classmethod
    @abc.abstractmethod
    def create_model(cls, bp):
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

    @classmethod
    def create_model(cls, bp):
        model_map = {
            'xgb': XgbModel
        }
        model_class = model_map[bp.model_class_name]
        return model_class(bp.name, bp.markers)

    def build(self):
        pass
