import abc
from collections import namedtuple

ModelBlueprint = namedtuple('ModelBlueprint',
                            ['name', 'markers', 'model_class_name', 'parent'])


class Schema(abc.ABC):
    def __init__(self, model_blueprints):
        self._ready = False
        self._model_dict = {None: {'model': None, 'children': []}}
        for bp in model_blueprints:
            model = self.create_model(bp)
            self.add_model(model)
        self.build()

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

    def build(self, level=0, models=None):
        if models is None:
            models = self._model_dict[None]['children'].copy()
        for name in models:
            model = self._model_dict[name]['model']
            model.level = level
            children = self._model_dict[name]['children']
            if len(children) > 0:
                self.build(level=level + 1, models=children)

    @classmethod
    @abc.abstractmethod
    def create_model(cls, bp):
        pass


class Classifier:
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

    def predict(self, x, data_frame_class, series_class):
        df = data_frame_class()
        for model in self.schema.walk():
            parent_level_label = 'level{}'.format(model.level - 1)

            if parent_level_label in df.columns:
                col = df.get_col_series(parent_level_label)
                index = col.get_item_index(model.name)
            else:
                index = x.index

            y = model.predict(x.loc(index))
            y = series_class(y, index=index)
            level_label = 'level{}'.format(model.level)
            df.set_col(level_label, y)
        return df


class AbsDataFrame(abc.ABC):
    @property
    @abc.abstractmethod
    def columns(self):
        pass

    @property
    @abc.abstractmethod
    def index(self):
        pass

    @property
    @abc.abstractmethod
    def values(self):
        pass

    @abc.abstractmethod
    def get_col_series(self, name, index=None):
        return AbsSeries()

    @abc.abstractmethod
    def loc(self, index=None, columns=None):
        return AbsDataFrame()

    @abc.abstractmethod
    def set_col(self, col_name, series):
        pass


class AbsSeries(abc.ABC):
    @property
    @abc.abstractmethod
    def index(self):
        pass

    @property
    @abc.abstractmethod
    def values(self):
        pass

    @abc.abstractmethod
    def get_item_index(self, item):
        pass

    @abc.abstractmethod
    def unique(self):
        pass


class DataMapper:
    def create_data_map(self, x_train, y_train):
        r = dict()
        blocks = [
            {'name': 'all-events',
             'index': y_train.index,
             'parent': None}
        ]
        for column in y_train.columns:
            new_blocks = list()
            for block in blocks:
                # x_train_block = x_train.loc(block['index'],
                # schema.get(block['name'])
                x_train_block = x_train.loc(block['index'])
                y_train_block = y_train.get_col_series(column,
                                                       index=block['index'])
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
            index = series_y.get_item_index(name)
            blocks.append({'name': name,
                           'index': index,
                           'parent': parent})
        return blocks


class Model(abc.ABC):
    def __init__(self, name, markers, parent, level=None, **kwargs):
        self.name = name
        self.markers = markers
        self.parent = parent
        self.level = level
        self._model_ins = self.init_model_instance(**kwargs)

    @abc.abstractmethod
    def init_model_instance(self, **kwargs):
        pass

    def filter_x(self, x):
        return x.loc(index=None, columns=self.markers)

    def fit(self, x_train, y_train):
        self._model_ins.fit(self.filter_x(x_train).values,
                            y_train.values)

    def score(self, x_train, y_train):
        return self._model_ins.score(self.filter_x(x_train).values,
                                     y_train.values)

    def predict(self, x):
        return self._model_ins.predict(self.filter_x(x).values)
