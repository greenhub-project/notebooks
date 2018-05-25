#!/usr/bin/env python3

import pickle
import pandas as pd


def average_mem_type(df):
    for dtype in ['float', 'int', 'object']:
        selected_dtype = df.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print("Average memory usage for {} columns: {:03.2f} MB".format(
            dtype, mean_usage_mb))


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def cache_dtypes(df, ignored=[]):
    dtypes = df.drop(ignored, axis=1).dtypes
    dtypes_col = dtypes.index
    dtypes_type = [i.name for i in dtypes.values]
    return dict(zip(dtypes_col, dtypes_type))


def save_dtypes(dtypes, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dtypes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(filename, 'created!')


def load_dtypes(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
