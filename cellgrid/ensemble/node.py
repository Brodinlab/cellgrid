class NodesTrainer:
    def __init__(self, schema):
        self.schema = schema
        self.nm = NodeManager()

    def schema_to_nodes(self):
        for row in self.schema.get():
            self.nm.add(row['name'], row['parent'],
                        row['model_class'], row['markers'])

    def update_node_data(self, x_train, y_train):
        blocks = [{'name': 'all-events', 'index': y_train.index, 'parent': None}]
        for column in y_train:
            new_blocks = list()
            for block in blocks:
                x_train_block = x_train.loc[block['index']]
                y_train_block = y_train.loc[block['index'], column]
                self.nm.update_node_data(block['name'], x_train_block, y_train_block)
                new_blocks += self.create_new_block(y_train_block, block['parent'])
            blocks = new_blocks

    def create_new_block(self, y_train, parent):
        series_y = y_train
        blocks = list()
        for name in series_y.unique():
            blocks.append({'name': name,
                           'index': series_y[series_y == name].index,
                           'parent': parent})
        return blocks

    def fit(self):
        pass


class NodeManager:
    def __init__(self):
        self.node_dict = {None: {'node': None, 'children': []}}

    def add(self, name, parent, model_class, markers):
        node = ModelNode(name, markers, model_class=model_class)
        self.node_dict[name] = {'node': node, 'children': []}
        self.node_dict[parent]['children'].append(name)

    def update_node_data(self, name, x_train, y_train):
        node = self.get_node(name)
        node.x_train = x_train[node.markers]
        node.y_train = y_train

    def get_node(self, name):
        return self.node_dict[name]['node']

    def walk(self):
        root = self.node_dict[None]
        nodes = root['children']
        for name in nodes:
            yield self.node_dict[name]['node']
            nodes.extend(self.node_dict[name]['children'])


class ModelNode:
    def __init__(self, name, markers, model_class=None,
                 model_ins=None, x_train=None, y_train=None):
        self._name = name
        self._markers = markers
        self._model_class = model_class
        self._model_ins = model_ins
        self._x_train = x_train
        self._y_train = y_train

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
