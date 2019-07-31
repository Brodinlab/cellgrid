import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics.scorer import check_scoring


class NodesTrainer:
    def __init__(self, schema):
        self.schema = schema
        self.nm = NodeManager()

    def schema_to_nodes(self):
        for row in self.schema.get():
            self.nm.add(row['name'], row['parent'],
                        row['model_class'], row['markers'])
        self.nm.build_level()

    def update_node_data(self, x_train, y_train):
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
                    self.nm.update_node_data(block['name'],
                                             x_train_block,
                                             y_train_block)
                new_blocks += self.create_new_block(y_train_block,
                                                    block['parent'])
            blocks = new_blocks

    @staticmethod
    def create_new_block(y_train, parent):
        series_y = y_train
        blocks = list()
        for name in series_y.unique():
            index = series_y[series_y == name].index
            blocks.append({'name': name,
                           'index': index,
                           'parent': parent})
        return blocks

    def fit(self):
        for node in self.nm.walk():
            node.fit()

    def score(self):
        r = dict()
        for node in self.nm.walk():
            r[node.name] = node.score()
        return r

    def predict(self, x):
        df = pd.DataFrame([])
        for node in self.nm.walk():
            parent_level_label = 'level{}'.format(node.level - 1)

            if parent_level_label in df:
                col = df[parent_level_label]
                index = col[col == node.name].index
            else:
                index = x.index

            y = pd.Series(node.predict(x.loc[index]),
                          index=index)

            level_label = 'level{}'.format(node.level)
            if index.shape == x.index.shape:
                df[level_label] = y
            else:
                df.loc[y.index, level_label] = y

        return df


class NodeManager:
    def __init__(self):
        self.node_dict = {None: {'node': None, 'children': []}}

    def add(self, name, parent, model_class, markers):
        if parent in self.node_dict:
            self.node_dict[parent]['children'].append(name)
        else:
            self.node_dict[parent] = {'node': None, 'children': [name]}

        node = ModelNode(name, markers, model_class=model_class)
        if name in self.node_dict:
            self.node_dict[name]['node'] = node
        else:
            self.node_dict[name] = {'node': node, 'children': []}

    def build_level(self, level=0, nodes=None):
        if nodes is None:
            nodes = self.node_dict[None]['children'].copy()
        for name in nodes:
            node = self.node_dict[name]['node']
            node.level = level
            children = self.node_dict[name]['children']
            if len(children) > 0:
                self.build_level(level=level + 1, nodes=children)

    def update_node_data(self, name, x_train, y_train):
        node = self.get_node(name)
        node.x_train = x_train[node.markers]
        node.y_train = y_train

    def get_node(self, name):
        return self.node_dict[name]['node']

    def walk(self):
        nodes = self.node_dict[None]['children'].copy()
        for name in nodes:
            yield self.node_dict[name]['node']
            nodes.extend(sorted(self.node_dict[name]['children']))


class ModelNode:
    def __init__(self, name, markers, model_class=None,
                 model_ins=None, x_train=None,
                 y_train=None, level=None):
        self._name = name
        self._markers = markers
        self._model_class = model_class
        self._model_ins = model_ins
        self._x_train = x_train
        self._y_train = y_train
        self._level = level

    @property
    def name(self):
        return self._name

    @property
    def markers(self):
        return self._markers

    @property
    def x_train(self):
        return self._x_train

    @x_train.setter
    def x_train(self, x_train):
        self._x_train = x_train

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, y_train):
        self._y_train = y_train

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = level

    def fit(self):
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

        model_class = model_map[self._model_class]['model_class']
        params = model_map[self._model_class]['params']
        self._model_ins = model_class(**params)
        self._model_ins.fit(self._x_train[self._markers], self._y_train)

    def score(self):
        return check_scoring(self._model_ins)(self._model_ins,
                                              self._x_train[self._markers],
                                              self._y_train)

    def predict(self, x):
        return self._model_ins.predict(x[self._markers])
