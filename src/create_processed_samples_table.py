import sys
import numpy as np
import pandas as pd
from utils import mem_usage, save_df, typecast_objects, typecast_ints, typecast_floats

def main():
    try:
        #if path isn't given (csv files are in the same directory as script)
        if len(sys.argv) == 1:
            cols = ['id', 'device_id', 'timestamp', 'battery_state', 'battery_level']
            df_samples = pd.read_csv('samples.csv', usecols=cols, parse_dates=['timestamp'])

            cols = ['sample_id', 'bluetooth_enabled', 'location_enabled', 'power_saver_enabled', 'flashlight_enabled', 'nfc_enabled', 'unknown_sources', 'developer_mode']
            df_settings = pd.read_csv('settings.csv', usecols=cols)

            df_samples.set_index('id', inplace=True)
            df_settings.set_index('sample_id', inplace=True)

            joined_samples = df_samples.join(df_settings)
        #if path is given (path to the csv files folder)
        elif len(sys.argv) == 2:
            cols = ['id', 'device_id', 'timestamp', 'battery_state', 'battery_level']
            df_samples = pd.read_csv(sys.argv[1] + '\samples.csv', usecols=cols, parse_dates=['timestamp'])

            cols = ['sample_id', 'bluetooth_enabled', 'location_enabled', 'power_saver_enabled', 'flashlight_enabled', 'nfc_enabled', 'unknown_sources', 'developer_mode']
            df_settings = pd.read_csv(sys.argv[1] + '\settings.csv', usecols=cols)

            df_samples.set_index('id', inplace=True)
            df_settings.set_index('sample_id', inplace=True)

            joined_samples = df_samples.join(df_settings)
        else:
            raise IOError('Dataset missing!')

        print('------>Before:<------')
        joined_samples.info(memory_usage='deep')

        # remove duplicate samples
        joined_samples.drop_duplicates(subset=['timestamp', 'battery_state', 'battery_level'], inplace=True)

        # sorting
        joined_samples = joined_samples.sort_values(by=['device_id', 'timestamp'])

        # filtering
        joined_samples = joined_samples[joined_samples.timestamp >= pd.Timestamp('2017-10-15')]

        # remove duplicate samples with diferent timestamps
        duplicated_samples = joined_samples.loc[joined_samples.battery_level.diff() == 0].index
        joined_samples.drop(duplicated_samples, inplace=True)

        # reset indexes
        joined_samples = joined_samples.reset_index(drop=True)

        # explicitly cast battery level to integer
        joined_samples['battery_level'] = joined_samples['battery_level'] * 100

        #compute facts ids and drop them from dataframe
        facts = ['bluetooth_enabled', 'location_enabled', 'power_saver_enabled', 'flashlight_enabled', 'nfc_enabled', 'unknown_sources', 'developer_mode']
        joined_samples.insert(1, 'service_comb_id', np.packbits(joined_samples[facts].values, axis=-1))
        joined_samples.drop(facts, axis=1, inplace=True)

        #creates a column with the average time per 1% for each row
        joined_samples['time_diff'] = joined_samples['timestamp'].diff().dt.total_seconds() / joined_samples['battery_level'].diff()
        full_rows = joined_samples[joined_samples.battery_state == 'Full']
        joined_samples.loc[joined_samples['battery_state'] != joined_samples['battery_state'].shift(), 'time_diff'] = None
        joined_samples[joined_samples.battery_state == 'Full'] = full_rows

        #separates samples between devices
        joined_samples.loc[joined_samples['device_id'] != joined_samples['device_id'].shift(), 'time_diff'] = None

        # ------------------------------downcast all types datfarame and crete binary file-----------------------------------
        # special cast battery level
        joined_samples['battery_level'] = joined_samples['battery_level'].astype(np.uint8)

        # downcast integer columns
        converted_int = typecast_ints(joined_samples.select_dtypes(include=['integer']))

        # downcast float columns
        converted_float = typecast_floats(joined_samples.select_dtypes(include=['float']))

        # convert object columns to lowercase
        joined_samples_obj = joined_samples.select_dtypes(include=['object'])
        joined_samples_obj = joined_samples_obj.apply(lambda x: x.str.lower())

        # convert object to category columns
        # when unique values < 50% of total
        converted_obj = typecast_objects(joined_samples_obj)

        # transform optimized types
        joined_samples[converted_int.columns] = converted_int
        joined_samples[converted_float.columns] = converted_float
        joined_samples[converted_obj.columns] = converted_obj

        print('------------------------------')
        print('------>After:<------')
        joined_samples.info(memory_usage='deep')
        print('------------------------------')

        save_df(joined_samples, 'processed_samples.parquet')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
