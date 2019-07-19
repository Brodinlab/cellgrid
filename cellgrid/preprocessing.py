from fcsy.fcs import read_fcs
from fcsy.preprocessing import arcsinh, zscore

DATA_MARKERS = ('CD45', 'HLA-ABC', 'CD57', 'CD19', 'CD5', 'CD16',
                'CD4', 'CD8a', 'CD11c', 'CD31', 'CD25', 'CD64',
                'CD123', 'gdTCR', 'CD13', 'CD3e', 'CD7', 'CD26',
                'CD9', 'CD22', 'CD14', 'CD161', 'CD29', 'HLA-DR',
                'CD44', 'CD127', 'CD24', 'CD27', 'CD38', 'CD45RA',
                'CD20', 'CD33', 'IgD', 'CD56', 'CD99', 'CD15', 'CD39',
                'CD11b')

DNA_BEADS_MARKERS = ('Ce140Di', 'Ir191Di')


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
