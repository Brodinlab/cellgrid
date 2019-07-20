from cellgrid.ensemble import *


class TestGridSchema:
    def test_from_json(self):
        schema = GridSchema.from_json()
        assert isinstance(schema, GridSchema)
        assert len(schema.get()) == 12
        assert schema.get()[0]['parent'] is None
