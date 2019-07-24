from cellgrid.ensemble.node import *


class TestNode:
    def setup_method(self):
        self.node = ModelNode('test', ['a', 'b'])

    def test_set_xtrain(self):
        self.node.x_train = 'x_train'
        assert self.node.x_train == 'x_train'

    def test_set_ytrain(self):
        self.node.y_train = 'y_train'
        assert self.node.y_train == 'y_train'


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

    def test_walk(self):
        nm = NodeManager()
        nm.add('test', None, 'model_class', 'markers')
        nm.add('test2', 'test', 'model_class', 'markers')
        nm.add('test3', 'test', 'model_class', 'markers')
        nm.add('test4', 'test3', 'model_class', 'markers')

        r = [i.name for i in nm.walk()]
        assert r == ['test', 'test2', 'test3', 'test4']



