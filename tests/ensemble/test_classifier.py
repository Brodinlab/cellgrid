import os
import pandas as pd
from cellgrid.ensemble import *


class TestGridSchema:
    def test_from_json(self):
        schema = GridSchema.from_json()
        assert isinstance(schema, GridSchema)
        assert len(schema.get()) == 8
        assert schema.get()[0]['parent'] is None

    def test_fit(self):
        f = os.path.join(os.getcwd(),
                         'tests', 'cellgrid_test.csv')
        df = pd.read_csv(f)
        y_train = df[['level0', 'level1', 'level2']]
        x_train = df.drop(['level0', 'level1', 'level2'], axis=1)
        schema = GridSchema.from_json()
        clf = GridClassifier(schema)
        clf.fit(x_train, y_train)
        r = clf.score(x_train, y_train)

        assert list(r.keys()) == ['all-events', 'cells', 'CD4T', 'B',
                                  'gdT', 'NK', 'CD8T', 'Monocytes']
        for v in r.values():
            assert v > 0.9
