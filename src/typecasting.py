#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from utils import mem_usage, cache_dtypes, save_dtypes


def main():
    try:
        if len(sys.argv) < 2:
            raise IOError('Dataset missing!')

        gl = pd.read_csv(sys.argv[1])
        gl.info(memory_usage='deep')

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
        optimized_gl = gl.copy()
        optimized_gl[converted_int.columns] = converted_int
        optimized_gl[converted_float.columns] = converted_float
        optimized_gl[converted_obj.columns] = converted_obj

        print('Before:', mem_usage(gl))
        print('After:', mem_usage(optimized_gl))

        fcache = sys.argv[1].split('.')[0] + '.pkl'
        save_dtypes(cache_dtypes(optimized_gl), fcache)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
