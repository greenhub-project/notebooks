import sys
import numpy as np
import pandas as pd
from utils import *


def main():
    try:
        if len(sys.argv) < 2:
            raise IOError('Dataset missing!')

        gl = pd.read_csv(sys.argv[1])
        print('Before:', mem_usage(gl))
        gl.info(memory_usage='deep')

        # downcast integer columns
        converted_int = typecast_ints(gl.select_dtypes(include=['int']))

        # downcast float columns
        converted_float = typecast_floats(gl.select_dtypes(include=['float']))

        # convert object columns to lowercase
        gl_obj = gl.select_dtypes(include=['object'])
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

        fname = sys.argv[1].split('.')[0]

        save_dtypes(cache_dtypes(gl), fname + '.pkl')
        save_df(gl, fname + '.parquet')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
