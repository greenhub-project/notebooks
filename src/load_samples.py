import numpy as np
import pandas as pd
from utils import mem_usage, typecast_objects, save_df


def main():
    try:
        cols = ['device_id', 'timestamp', 'battery_state', 'battery_level',
                'network_status', 'screen_brightness', 'screen_on']

        samples = pd.read_csv('samples.csv', usecols=cols,
                              parse_dates=['timestamp'])

        cols = ['bluetooth_enabled', 'location_enabled', 'power_saver_enabled',
                'nfc_enabled', 'unknown_sources', 'developer_mode']

        settings = pd.read_csv('settings.csv', usecols=cols)

        gl = samples.join(settings)

        print('Before:', mem_usage(gl))

        # sorting
        gl = gl.sort_values(by=['device_id', 'timestamp'])

        # filtering
        gl = gl[pd.Timestamp('2017-10-15') <= gl.timestamp]

        # reset indexes
        gl = gl.reset_index(drop=True)

        # explicitly cast battery level to integer
        gl_level = gl.battery_level * 100
        converted_level = gl_level.astype(np.uint8)

        # downcast integer columns
        gl_int = gl.select_dtypes(include=['int'])
        converted_int = gl_int.apply(pd.to_numeric, downcast='unsigned')

        # downcast float columns
        gl_float = gl.select_dtypes(
            include=['float']).drop('battery_level', axis=1)
        converted_float = gl_float.apply(pd.to_numeric, downcast='float')

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
        gl['battery_level'] = converted_level

        print('After:', mem_usage(gl))
        gl.info(memory_usage='deep')

        save_df(gl, 'samples.parquet')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
