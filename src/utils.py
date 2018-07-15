#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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


def downcastDfTypes(df):
    # downcast integer columns
    converted_int = typecast_ints(df.select_dtypes(include=['integer']))

    # downcast float columns
    converted_float = typecast_floats(df.select_dtypes(include=['float']))

    # convert object columns to lowercase
    df_obj = df.select_dtypes(include=['object'])
    df_obj = df_obj.apply(lambda x: x.str.lower())

    # convert object to category columns
    # when unique values < 50% of total
    converted_obj = typecast_objects(df_obj)

    # transform optimized types
    df[converted_int.columns] = converted_int
    df[converted_float.columns] = converted_float
    df[converted_obj.columns] = converted_obj

    return df


def save_df(df, path, compression='snappy', use_dictionary=True):
    """
    Save a pandas DataFrame to a parquet file
    """
    try:
        print('Creating parquet file...')
        df.to_parquet(path, compression=compression, use_dictionary=use_dictionary)
        print(path, 'created!')
    except Exception as e:
        print(e)


def load_df(path, columns=None, nthreads=4, strings_to_categorical=True):
    """
    Load a parquet file and returns a pandas DataFrame
    """
    try:
        table = pq.read_table(path, columns=columns, nthreads=nthreads)
        return table.to_pandas(strings_to_categorical=strings_to_categorical)
    except Exception as e:
        print(e)
