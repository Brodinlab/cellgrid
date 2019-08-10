import pandas as pd
from pandas.util.testing import assert_series_equal, assert_frame_equal
from cellgrid.core import Schema, ModelBlueprint, DataMapper
from cellgrid.ensemble.classifier import DataFrame, Series


class ModeTestClass:
    def __init__(self, bp):
        self.name = bp.name
        self.parent = bp.parent


class SchemaTestClass(Schema):
    @classmethod
    def create_model(cls, bp):
        return ModeTestClass(bp)


class TestSchema:
    def test_add_node(self):
        schema = SchemaTestClass([])
        bp = ModelBlueprint('test', 'markers', 'xgb', None)
        xgb = SchemaTestClass.create_model(bp)
        schema.add_model(xgb)
        assert xgb == schema._model_dict['test']['model']

        bp2 = ModelBlueprint('test2', 'markers', 'xgb', 'test')
        xgb2 = SchemaTestClass.create_model(bp2)
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
        schema = SchemaTestClass(bps)
        r = [(i.name, i.level) for i in schema.walk()]
        assert r == [('test0', 0),
                     ('test11', 1),
                     ('test12', 1),
                     ('test111', 2),
                     ('test112', 2),
                     ('test121', 2),
                     ]


class TestDataFrameAndSeries:
    def test_get_col_series(self):
        df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=list('ab'))
        df = DataFrame(df)
        s = df.get_col_series('b')
        assert_series_equal(s.s, pd.Series([2, 4, 6], name='b'))
        assert isinstance(s, Series)

        s2 = df.get_col_series('a', index=[1])
        assert_series_equal(s2.s, pd.Series([3], name='a', index=[1]))

    def test_loc(self):
        df_pd = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=list('ab'))
        df = DataFrame(df_pd)

        df_loc = df.loc(index=[0, 1])
        assert_frame_equal(df_loc.df, pd.DataFrame([[1, 2], [3, 4]], columns=list('ab')))
        assert isinstance(df_loc, DataFrame)

        df_loc2 = df.loc()
        assert_frame_equal(df_loc2.df, df_pd)
        assert isinstance(df_loc2, DataFrame)

        df_loc3 = df.loc(index=[0, 2], columns=['a'])
        assert_frame_equal(df_loc3.df, pd.DataFrame([[1], [5]], index=[0, 2], columns=['a']))
        assert isinstance(df_loc3, DataFrame)

        df_loc4 = df.loc(columns=['b'])
        assert_frame_equal(df_loc4.df, pd.DataFrame([[2], [4], [6]], columns=['b']))
        assert isinstance(df_loc4, DataFrame)

    def test_empty_df_set_col(self):
        df = DataFrame()
        s = Series(([11, 13]), name='x', index=[0, 2])
        df.set_col('a', s)
        assert_frame_equal(df.df, pd.DataFrame([[11], [13]], columns=['a'], index=[0, 2]))

    def test_set_col(self):
        df_pd = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=list('ab'))
        df = DataFrame(df_pd)
        s_pd = pd.Series(([11, 13]), name='x', index=[0, 1])
        s = Series(s_pd.values, index=s_pd.index, name=s_pd.name)
        df.set_col('a', s)
        assert_frame_equal(df.df,
                           pd.DataFrame([[11, 2], [13, 4], [5, 6]], columns=list('ab')))

        s_pd2 = pd.Series(([16]), name='a', index=[2])
        s2 = Series(s_pd2.values, index=s_pd2.index, name=s_pd.name)
        df.set_col('b', s2)
        assert_frame_equal(df.df,
                           pd.DataFrame([[11, 2], [13, 4], [5, 16]], columns=list('ab')))


class TestDataMapper:
    def test_create_data_map(self):
        x_train = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]],
                               columns=list('ab'))
        y_train = pd.DataFrame([['n1', 'n11'], ['n2', ''],
                                ['n2', ''], ['n1', 'n12']], columns=['x1', 'x2'])
        dm = DataMapper()
        data_map = dm.create_data_map(DataFrame(x_train), DataFrame(y_train))

        assert_frame_equal(data_map['all-events'][0].df, x_train)
        assert_series_equal(data_map['all-events'][1].s,
                            pd.Series(['n1', 'n2', 'n2', 'n1'], name='x1'))
        assert_frame_equal(data_map['n1'][0].df,
                           pd.DataFrame([[1, 2], [7, 8]], index=[0, 3], columns=list('ab'))
                           )
        assert_series_equal(data_map['n1'][1].s,
                            pd.Series(['n11', 'n12'], name='x2', index=[0, 3])
                            )

