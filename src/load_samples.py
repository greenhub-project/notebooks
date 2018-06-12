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

        # filtering
        samples_df = samples_df[samples_df.timestamp >= pd.Timestamp('2016-1-1')]

        # remove duplicate samples
        samples_df.drop_duplicates(subset=['timestamp', 'battery_state', 'battery_level'], inplace=True)

        # sorting
        samples_df = samples_df.sort_values(by=['device_id', 'timestamp'])

        # reset indexes
        samples_df = samples_df.reset_index(drop=True)

        # change battery level to int
        samples_df['battery_level'] = samples_df['battery_level'] * 100

        # add column with service combination ids
        facts = ['bluetooth_enabled', 'location_enabled', 'power_saver_enabled',
                 'flashlight_enabled', 'nfc_enabled', 'unknown_sources', 'developer_mode']

        samples_df['service_comb'] = np.packbits(samples_df[facts].values, axis=-1)

        # add column with average time to charge 1% for each row
        samples_df['time_diff'] = (samples_df['timestamp'].diff().dt.total_seconds()) / abs(samples_df['battery_level'].diff())
        samples_df.loc[samples_df['battery_state'] != samples_df['battery_state'].shift(), 'time_diff'] = None
        samples_df.loc[np.isinf(samples_df['time_diff']), 'time_diff'] = None

        #convert battery level to int
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
