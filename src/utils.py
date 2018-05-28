#!/usr/bin/env python3

import pickle
import pandas as pd
import pyarrow.parquet as pq


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


def save_dtypes(dtypes, path):
    with open(path, 'wb') as handle:
        pickle.dump(dtypes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(path, 'created!')


def load_dtypes(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def typecast_ints(gl_int):
    return gl_int.apply(pd.to_numeric, downcast='unsigned')


def typecast_floats(gl_float):
    return gl_float.apply(pd.to_numeric, downcast='float')


def typecast_objects(gl_obj):
    # convert object to category columns
    # when unique values < 50% of total
    converted_obj = pd.DataFrame()
    for col in gl_obj.columns:
        num_unique_values = len(gl_obj[col].unique())
        num_total_values = len(gl_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:, col] = gl_obj[col].astype('category')
        else:
            converted_obj.loc[:, col] = gl_obj[col]
    return converted_obj


def save_df(df, path, compression='snappy', use_dictionary=True):
    try:
        df.to_parquet(path, compression=compression,
                      use_dictionary=use_dictionary)
        print(path, 'created!')
    except Exception as e:
        print(e)


def load_df(path, columns=None, nthreads=4, strings_to_categorical=True):
    try:
        table = pq.read_table(path, columns=columns, nthreads=nthreads)
        return table.to_pandas(strings_to_categorical=strings_to_categorical)
    except Exception as e:
        print(e)
