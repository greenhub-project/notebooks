#!/usr/bin/env python3

import sys
import pandas as pd
from utils import save_df

CHUNKSIZE = 2500000


def typecast_chunk(gl):
    # downcast integer columns
    gl_int = gl.select_dtypes(include=['int'])
    converted_int = gl_int.apply(pd.to_numeric, downcast='unsigned')

    # downcast float columns
    gl_float = gl.select_dtypes(include=['float'])
    converted_float = gl_float.apply(pd.to_numeric, downcast='float')

    # convert object columns to lowercase
    gl_obj = gl.select_dtypes(include=['object'])
    gl_obj = gl_obj.apply(lambda x: x.str.lower())

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

    # transform optimized types
    gl[converted_int.columns] = converted_int
    gl[converted_float.columns] = converted_float
    gl[converted_obj.columns] = converted_obj

    return gl


def main():
    try:
        if len(sys.argv) < 2:
            raise IOError('Limit arg missing!')

        limit = int(sys.argv[1]) - 1

        chunks = pd.read_csv('app_processes.csv', usecols=[
                             'sample_id', 'name', 'is_system_app'], chunksize=CHUNKSIZE)

        for i, chunk in enumerate(chunks):
            filename = 'apps/processes.chunk' + str(i) + '.parquet'
            save_df(typecast_chunk(chunk), filename)
            if i > 0 and i == limit:
                break

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
