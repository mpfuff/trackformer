from os import listdir

import json
import pickle
import os, errno

import pandas as pd
from numpy import array
from pandas import DataFrame
from typing import cast
import io
from pathlib import Path


class MpFileUtil:

    def save_pickle(self, dir_name: str, file_name: str, obj: object):
        self.make_dir(dir_name)
        path = os.path.join(dir_name, file_name)
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, dir_name: str, file_name: str) -> object:
        path = os.path.join(dir_name, file_name)
        with open(path, 'rb') as f:
            meta_data = pickle.load(f)
        return meta_data

    def save_to_hd5_store(self, store: pd.HDFStore, store_name: str, obj: object) -> str:
        store[store_name] = obj
        store.close()
        return store_name

    def save_hd5_to_new_store(self, file_name: str, store_name: str, obj: object) -> (pd.HDFStore, str):
        store = pd.HDFStore(file_name)
        store[store_name] = obj
        store.close()
        return store, store_name

    def load_hd5(self, store_name: str, store: pd.HDFStore) -> object:
        obj = store[store_name]
        store.close()
        return obj

    def df_to_csv(self, file_name: str, df: DataFrame, compression: bool = False):
        if compression:
            compression_opts = dict(method='zip',
                                    archive_name=file_name + '.csv')
            df.to_csv(file_name + '.zip', index=False,
                      compression=compression_opts)
        else:
            df.to_csv(file_name + '.csv', index=False)

    # conda install openpyxl
    def df_to_exel(self, file_name: str, df: DataFrame):
        df.to_excel(file_name + ".xlsx", sheet_name='dataframe')

    def close_stores(self, stores: list):
        for store in stores:
            if isinstance(store, pd.HDFStore):
                store = cast(pd.HDFStore, store)
                if store.is_open:
                    print('save closing store:', store.filename)
                    store.close()

    def write_tsv_files_np(self, strings: list, weights: array, dir: str = 'output',
                           metadata_filename: str = 'metadata.tsv',
                           vectors_filename: str = 'vectors.tsv'):

        vecs = self.to_path(dir, vectors_filename)
        meta = self.to_path(dir, metadata_filename)
        out_v = io.open(vecs, 'w', encoding='utf-8')
        out_m = io.open(meta, 'w', encoding='utf-8')

        for text, vec in zip(strings, weights):
            out_m.write(text + "\n")
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")

        out_v.close()
        out_m.close()

    def write_metadata_tsv_file(self, tokenizer_items, dir: str = 'output',
                                metadata_filename: str = 'metadata.tsv', add_unknown: bool = True):

        meta = self.to_path(dir, metadata_filename)
        out_m = io.open(meta, 'w', encoding='utf-8')

        # add 1 entry for "unknown" words in Embedding Layer
        if add_unknown:
            out_m.write("Z\n")

        for item in tokenizer_items:
            out_m.write(item[0] + "\n")

        out_m.close()
        return True

    def read_from_csv(self, path: str):
        csv = pd.read_csv(path, na_values='None')
        return csv

    def close_file_handle(self, fh: io.TextIOWrapper):
        fh.close()

    def delete_files(self, files: list, absolute: bool = True):
        for file in files:
            if absolute:
                os.system('rm -r ' + file)
            else:
                os.system('rm -r ./' + file)
        return True

    def silentremove(self, filenames: list):
        for filename in filenames:
            try:
                os.remove('./' + filename)
            except OSError as e:  # this would be "except OSError, e:" before Python 2.6
                if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or
                    raise

        return True

    def make_dir(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        return dirname

    def to_path(self, dirname, filename):
        os.makedirs(dirname, exist_ok=True)
        return os.path.join(dirname, filename)

    def is_file(self, dirname, filename):
        path = os.path.join(dirname, filename)
        my_file = Path(path)
        return my_file.is_file()

    def is_dir(self, dirname):
        my_file = Path(dirname)
        return my_file.is_dir()

    def write_to_file(self, inputs: str = '', directory: str = 'data', filename: str = 'file.csv'):
        file_path = self.to_path(directory, filename)
        out_path = io.open(file_path, 'w', encoding='utf-8')
        out_path.write(inputs)
        out_path.close()

    def load_json_file(self, dirname, filename):
        filepath = self.to_path(dirname, filename)
        json1_file = open(filepath)
        json1_str = json1_file.read()
        json1_data = json.loads(json1_str)
        return json1_data

    def list_all_files_in_dir(self, dirname):
        onlyfiles = [f for f in listdir(dirname) if self.is_file(dirname, f)]
        return onlyfiles

