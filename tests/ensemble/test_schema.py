from cellgrid.ensemble import GridSchema


class TestSchema:
    def test_from_json(self):
        schema = GridSchema.from_json()
        assert isinstance(schema, GridSchema)
        # assert len(schema.get()) == 8
        # assert schema.get()[0]['parent'] is None
