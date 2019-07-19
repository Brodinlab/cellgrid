import pandas as pd
from pandas.util.testing import assert_almost_equal, assert_frame_equal
from fcsy.preprocessing import arcsinh, zscore
from cellgrid.preprocessing import transform, FileTransformer


def test_transform():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=list('ab'))
    expect = zscore(arcsinh(df))
    assert_almost_equal(transform(df), expect)


class TestFileTransformer:
    def setup_method(self):

        def read_file_func(f, name_type):
            df = pd.DataFrame([[1, 2], [3, 4]])
            if name_type == 'short':
                df.columns = list('ab')
            elif name_type == 'long':
                df.columns = list('AB')
            return df

        def transform_func(df):
            return df + 1

        self.ft = FileTransformer('fake',
                                  dna_beads_columns=['A'],
                                  data_columns=['b'],
                                  transform_func=transform_func,
                                  read_file_func=read_file_func
                                  )

    def test_get_dna_beads_columns(self):
        expect = pd.DataFrame([[2], [4]], columns=['A'])
        assert_frame_equal(self.ft.get_dna_beads_columns(), expect)

    def test_get_data_columns(self):
        expect = pd.DataFrame([[3], [5]], columns=['b'])
        assert_frame_equal(self.ft.get_data_columns(), expect)
