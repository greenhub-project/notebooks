import numpy as np
import pandas as pd
from utils import mem_usage, typecast_objects, save_df


def main():
    try:
        cols = ['device_id', 'timestamp', 'battery_state', 'battery_level',
                'network_status', 'screen_brightness', 'screen_on']

        samples = pd.read_csv('samples.csv', usecols=cols,
                              parse_dates=['timestamp'])

        cols = ['voltage', 'temperature']

        battery_details = pd.read_csv('battery_details.csv', usecols=cols)

        samples = samples.join(battery_details)

        cols = ['bluetooth_enabled', 'location_enabled', 'power_saver_enabled',
                'nfc_enabled', 'unknown_sources', 'developer_mode']

        settings = pd.read_csv('settings.csv', usecols=cols)

        gl = samples.join(settings).copy()

        del [samples, battery_details, settings]

        print('Before:', mem_usage(gl))
        gl.info(memory_usage='deep')

        cols = ['device_id', 'timestamp', 'battery_state', 'battery_level',
                'network_status', 'screen_brightness', 'voltage', 'temperature',
                'screen_on', 'bluetooth_enabled', 'location_enabled', 'power_saver_enabled',
                'nfc_enabled', 'unknown_sources', 'developer_mode']

        # reorder columns
        gl = gl[cols]

        # sorting
        gl = gl.sort_values(by=['device_id', 'timestamp'])

        # date filtering
        gl = gl[pd.Timestamp('2017-10-15') <= gl.timestamp]

        # reset indexes
        gl = gl.reset_index(drop=True)

        # explicitly cast battery level to integer
        gl_level = gl.battery_level * 100
        converted_level = gl_level.astype(np.uint8)

        # convert values to correct unit and round them
        gl.loc[gl.voltage > 10, 'voltage'] = gl.voltage / 1000
        gl['voltage'] = gl.voltage.round(2)
        gl['temperature'] = gl.temperature.round(2)

        # downcast integer columns
        gl_int = gl.select_dtypes(include=['int'])
        converted_int = gl_int.apply(pd.to_numeric, downcast='unsigned')

        # downcast float columns
        gl_float = gl.select_dtypes(
            include=['float']).drop('battery_level', axis=1)
        converted_float = gl_float.apply(pd.to_numeric, downcast='float')

        # convert object to category columns
        # when unique values < 50% of total
        gl_obj = gl.select_dtypes(include=['object'])
        converted_obj = typecast_objects(gl_obj)

        # transform optimized types
        gl[converted_int.columns] = converted_int
        gl[converted_float.columns] = converted_float
        gl[converted_obj.columns] = converted_obj
        gl['battery_level'] = converted_level

        # filter out malformed records
        gl = gl[gl.battery_level <= 100]
        gl = gl[gl.temperature >= 0]

        print('After:', mem_usage(gl))
        gl.info(memory_usage='deep')

        save_df(gl, 'samples.parquet')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
