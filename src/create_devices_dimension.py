import sys
import numpy as np
import pandas as pd
from utils import load_df, mem_usage, save_df, typecast_objects, typecast_ints, typecast_floats


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


def main():
    try:
        #if path isn't given (csv files are in the same directory as script)
        if len(sys.argv) == 1:
            cols = ['id', 'model', 'manufacturer', 'brand', 'product', 'os_version', 'kernel_version', 'is_root']
            devices_df = pd.read_csv('devices.csv', usecols=cols)

        #if path is given (path to the csv files folder)
        elif len(sys.argv) == 2:
            cols = ['id', 'model', 'manufacturer', 'brand', 'product', 'os_version', 'kernel_version', 'is_root']
            devices_df = pd.read_csv(sys.argv[1] + '\devices.csv', usecols=cols)

        else:
            raise IOError('Dataset missing!')

        devices_df = devices_df.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

        devices_df = downcastDfTypes(devices_df)
        devices_df.info(memory_usage='deep')
        
        save_df(devices_df, 'devices_dimension.parquet')

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
