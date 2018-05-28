#!/usr/bin/env python3

import pandas as pd
from utils import load_df


def main():
    try:
        cols = None

        df = load_df('samples.parquet', cols).sort_values(
            by=['device_id', 'timestamp'])

        df = df.reset_index(drop=True)

        df_int = df.select_dtypes(include=['int'])
        converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')

        df[converted_int.columns] = converted_int

        # additional features
        df['auto_brightness'] = df.screen_brightness == -1
        df['time_diff'] = df['timestamp'].diff()
        df.loc[df.device_id != df.device_id.shift(), 'time_diff'] = None

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
