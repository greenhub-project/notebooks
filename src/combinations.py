import numpy as np
import pandas as pd
from utils import load_df, save_df


def main():
    try:
        cols = None

        df = load_df('samples.parquet', cols)

        df = df.reset_index(drop=True)

        # additional features
        df['wifi_enabled'] = (df['network_status'] == 'wifi').astype(np.uint8)
        df['time_diff'] = df['timestamp'].diff()
        df.loc[df.device_id != df.device_id.shift(), 'time_diff'] = None
        df['time_diff'] = df.time_diff.dt.total_seconds()
        df['discharging'] = df.battery_level.shift() >= df.battery_level
        df.loc[df.device_id != df.device_id.shift(), 'discharging'] = None

        # facts indexing
        facts = ['screen_on', 'bluetooth_enabled', 'location_enabled', 'power_saver_enabled',
                 'nfc_enabled', 'unknown_sources', 'developer_mode', 'wifi_enabled']

        df['combination'] = np.packbits(df[facts].values, axis=-1)

        # output columns
        cols = ['device_id', 'timestamp', 'battery_state',
                'battery_level', 'network_status', 'time_diff', 'combination']

        save_df(df[cols], 'combinations.parquet')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
