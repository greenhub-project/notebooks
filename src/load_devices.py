import sys
import numpy as np
import pandas as pd
from utils import mem_usage, save_dtypes, cache_dtypes, save_df, \
    typecast_ints, typecast_floats, typecast_objects


def main():
    try:
        cols = ['id', 'model', 'manufacturer', 'brand', 'product', 
                'os_version', 'kernel_version', 'is_root']

        gl = pd.read_csv('devices.csv', usecols=cols)
        print('Before:', mem_usage(gl))
        gl.info(memory_usage='deep')

        gl['os_version'] = gl.os_version.str.replace('\\n', '')

        # downcast integer columns
        converted_int = typecast_ints(gl.select_dtypes(include=['int']))

        # downcast float columns
        converted_float = typecast_floats(gl.select_dtypes(include=['float']))

        # convert object columns to lowercase
        gl_obj = gl.select_dtypes(include=['object'])
        gl_obj = gl_obj.apply(lambda x: x.str.strip())
        gl_obj = gl_obj.apply(lambda x: x.str.lower())

        # convert object to category columns
        # when unique values < 50% of total
        converted_obj = typecast_objects(gl_obj)

        # transform optimized types
        gl[converted_int.columns] = converted_int
        gl[converted_float.columns] = converted_float
        gl[converted_obj.columns] = converted_obj

        print('\nAfter:', mem_usage(gl))
        gl.info(memory_usage='deep')

        save_dtypes(cache_dtypes(gl), 'devices.pkl')
        save_df(gl, 'devices.parquet')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
