from cellgrid.ensemble import ModelBlueprint
from cellgrid.ensemble.schema import GridSchema
from cellgrid.ensemble.model import XgbModel


class TestSchema:
    # def test_from_json(self):
    #    schema = GridSchema.from_json()
    #    assert isinstance(schema, GridSchema)
    # assert len(schema.get()) == 8
    # assert schema.get()[0]['parent'] is None

    def test_create_model(self):
        bp = ModelBlueprint('test', 'markers', 'xgb', None)
        xgb = GridSchema.create_model(bp)
        assert isinstance(xgb, XgbModel)
        assert xgb._model_ins.n_jobs == 10
        assert xgb._model_ins.max_depth == 10
        assert xgb._model_ins.n_estimators == 40

    def test_add_node(self):
        schema = GridSchema([])
        bp = ModelBlueprint('test', 'markers', 'xgb', None)
        xgb = GridSchema.create_model(bp)
        schema.add_model(xgb)
        assert xgb == schema._model_dict['test']['model']

        bp2 = ModelBlueprint('test2', 'markers', 'xgb', 'test')
        xgb2 = GridSchema.create_model(bp2)
        schema.add_model(xgb2)
        assert schema._model_dict['test']['children'] == ['test2']

    def test_build_and_walk(self):
        bps = [
            ModelBlueprint('test11', 'markers', 'xgb', 'test0'),
            ModelBlueprint('test0', 'markers', 'xgb', None),
            ModelBlueprint('test111', 'markers', 'xgb', 'test11'),
            ModelBlueprint('test12', 'markers', 'xgb', 'test0'),
            ModelBlueprint('test112', 'markers', 'xgb', 'test11'),
            ModelBlueprint('test121', 'markers', 'xgb', 'test12')
        ]
        schema = GridSchema(bps)
        r = [(i.name, i.level) for i in schema.walk()]
        assert r == [('test0', 0),
                     ('test11', 1),
                     ('test12', 1),
                     ('test111', 2),
                     ('test112', 2),
                     ('test121', 2),
                     ]
