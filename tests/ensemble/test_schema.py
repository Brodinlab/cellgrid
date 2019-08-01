from cellgrid.ensemble import GridSchema, ModelBlueprint
from cellgrid.ensemble.model import XgbModel


class TestSchema:
    def test_from_json(self):
        schema = GridSchema.from_json()
        assert isinstance(schema, GridSchema)
        # assert len(schema.get()) == 8
        # assert schema.get()[0]['parent'] is None

    def test_create_model(self):
        bp = ModelBlueprint('test', 'markers', 'xgb', None)
        xgb = GridSchema.create_model(bp)
        assert isinstance(xgb, XgbModel)
