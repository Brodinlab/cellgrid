import pandas as pd
from xgboost import XGBClassifier
from pandas.util.testing import assert_index_equal, assert_series_equal
from cellgrid.ensemble.node import *
from cellgrid.ensemble import Schema


class TestNode:
    def setup_method(self):
        self.x_train = pd.DataFrame([[1, 2, 3],
                                     [4, 5, 6],
                                     [7, 8, 9]],
                                    columns=list('abc'))
        self.y_train = pd.Series(['a', 'a', 'b'])
        self.node = ModelNode('test', ['a', 'b'], model_class='xgb',
                              x_train=self.x_train, y_train=self.y_train)

    def test_set_xtrain(self):
        self.node.x_train = 'x_train'
        assert self.node.x_train == 'x_train'

    def test_set_ytrain(self):
        self.node.y_train = 'y_train'
        assert self.node.y_train == 'y_train'

    def test_fit(self):
        self.node.fit()
        assert isinstance(self.node._model_ins, XGBClassifier)


class TestNodeManager:
    def test_add(self):
        nm = NodeManager()
        nm.add('test', None, 'model_class', 'markers')
        node = nm.get_node('test')
        assert isinstance(node, ModelNode)
        assert node.name == 'test'

        nm.add('test2', 'test', 'model_class', 'markers')
        node2 = nm.get_node('test2')
        assert node2.name == 'test2'

    def test_build_level_and_walk(self):
        nm = NodeManager()
        nm.add('test11', 'test0', 'model_class', 'markers')
        nm.add('test0', None, 'model_class', 'markers')
        nm.add('test111', 'test11', 'model_class', 'markers')
        nm.add('test12', 'test0', 'model_class', 'markers')
        nm.add('test112', 'test11', 'model_class', 'markers')
        nm.add('test121', 'test12', 'model_class', 'markers')
        nm.build_level()
        r = [(i.name, i.level) for i in nm.walk()]
        assert r == [('test0', 0),
                     ('test11', 1),
                     ('test12', 1),
                     ('test111', 2),
                     ('test112', 2),
                     ('test121', 2),
                     ]


class TestNodeTrainer:
    def test_create_new_blocks(self):
        y_train = pd.Series([1, 2, 1])
        parent = 'parent'
        blocks = NodesTrainer.create_new_block(y_train, parent)

        assert blocks[0]['name'] == 1
        assert blocks[0]['parent'] == parent
        assert_index_equal(blocks[0]['index'], pd.Index([0, 2]))

        assert blocks[1]['name'] == 2
        assert blocks[1]['parent'] == parent
        assert_index_equal(blocks[1]['index'], pd.Index([1]))

    def test_update_node_data(self):
        schema_data = [
            {
                'name': 'all-events',
                'parent': None,
                'model_class': 'xgb',
                'markers': ['a', 'b']
            },
            {
                'name': 'n1',
                'parent': 'all-events',
                'model_class': 'xgb',
                'markers': ['a', 'b']
            }
        ]
        schema = Schema(schema_data)
        nt = NodesTrainer(schema)
        nt.schema_to_nodes()

        x_train = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]],
                               columns=list('ab'))
        y_train = pd.DataFrame([['n1', 'n11'], ['n2', ''], ['n2', ''], ['n1', 'n12']], columns=['x1', 'x2'])
        nt.update_node_data(x_train, y_train)

        nodes = list(nt.nm.walk())
        assert_series_equal(nodes[0].y_train,
                            pd.Series(['n1', 'n2', 'n2', 'n1'], name='x1'))
        assert_series_equal(nodes[1].y_train,
                            pd.Series(['n11', 'n12'], name='x2', index=[0, 3])
                            )
