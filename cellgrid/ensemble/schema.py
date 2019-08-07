import os
import json
from .classifier import ModelBlueprint, Schema
from .model import XgbModel


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
        models = self._model_dict[None]['children'].copy()
        for name in models:
            yield self._model_dict[name]['model']
            models.extend(sorted(self._model_dict[name]['children']))

    def add_model(self, model):
        if model.parent in self._model_dict:
            self._model_dict[model.parent]['children'].append(model.name)
        else:
            self._model_dict[model.parent] = {'node': None,
                                              'children': [model.name]}

        if model.name in self._model_dict:
            self._model_dict[model.name]['model'] = model
        else:
            self._model_dict[model.name] = {'model': model, 'children': []}

    @classmethod
    def create_model(cls, bp):
        model_map = {
            'xgb': XgbModel
        }
        model_class = model_map[bp.model_class_name]
        return model_class(bp.name, bp.markers, bp.parent)

    def build(self, level=0, models=None):
        if models is None:
            models = self._model_dict[None]['children'].copy()
        for name in models:
            model = self._model_dict[name]['model']
            model.level = level
            children = self._model_dict[name]['children']
            if len(children) > 0:
                self.build(level=level + 1, models=children)
