import sys
import numpy as np
import pandas as pd
from utils import mem_usage, save_df, typecast_objects, typecast_ints, typecast_floats

def main():
    try:
        if len(sys.argv) < 2:
            raise IOError('Dataset missing!')

        cols = ['device_id', 'timestamp', 'battery_state', 'battery_level', 'bluetooth_enabled', 'location_enabled',
                'power_saver_enabled', 'flashlight_enabled', 'nfc_enabled', 'unknown_sources', 'developer_mode']

        samples_df = pd.read_csv(sys.argv[1], usecols=cols, parse_dates=['timestamp'])
        print('Before:', mem_usage(samples_df))
        print(samples_df.dtypes)

        # remove duplicate samples
        samples_df.drop_duplicates(subset=['timestamp', 'battery_state', 'battery_level'], inplace=True)

        # sorting
        samples_df = samples_df.sort_values(by=['device_id', 'timestamp'])

        # filtering
        samples_df = samples_df[samples_df.timestamp >= pd.Timestamp('2016-1-1')]

        # reset indexes
        samples_df = samples_df.reset_index(drop=True)

        # explicitly cast battery level to integer
        samples_df['battery_level'] = samples_df['battery_level'] * 100
        samples_df['battery_level'] = samples_df['battery_level'].astype(np.uint8)

        # downcast integer columns
        converted_int = typecast_ints(samples_df.select_dtypes(include=['integer']))

        # downcast float columns
        converted_float = typecast_floats(samples_df.select_dtypes(include=['float']))

        # convert object columns to lowercase
        samples_df_obj = samples_df.select_dtypes(include=['object'])
        samples_df_obj = samples_df_obj.apply(lambda x: x.str.lower())

        # convert object to category columns
        # when unique values < 50% of total
        converted_obj = typecast_objects(samples_df_obj)

        # transform optimized types
        samples_df[converted_int.columns] = converted_int
        samples_df[converted_float.columns] = converted_float
        samples_df[converted_obj.columns] = converted_obj

        print('------------------------------')
        print('After:', mem_usage(samples_df))
        print(samples_df.dtypes)
        print('------------------------------')

        save_df(samples_df, 'samples.parquet')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
