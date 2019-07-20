from fcsy.fcs import read_fcs
from fcsy.preprocessing import arcsinh, zscore
from .const import DATA_MARKERS, DNA_BEADS_MARKERS


def transform(df):
    df = arcsinh(df)
    df = zscore(df)
    return df


class FileTransformer:
    def __init__(self,
                 filepath,
                 dna_beads_columns=DNA_BEADS_MARKERS,
                 data_columns=DATA_MARKERS,
                 transform_func=transform,
                 read_file_func=read_fcs
                 ):
        self.filepath = filepath
        self.dna_beads_columns = dna_beads_columns
        self.data_columns = data_columns
        self.transform_func = transform_func
        self.read_file_func = read_file_func

    def get_dna_beads_columns(self):
        df = self.read_file_func(self.filepath, name_type='long')
        return self.transform_func(df[self.dna_beads_columns])

    def get_data_columns(self):
        df = self.read_file_func(self.filepath, name_type='short')
        return self.transform_func(df[self.data_columns])
