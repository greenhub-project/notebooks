import sys
import numpy as np
import pandas as pd
from utils import load_df, save_df

def main():
    try:
        if len(sys.argv) < 2:
            raise IOError('Dataset missing!')

        cols = ['bluetooth_enabled', 'location_enabled', 'power_saver_enabled', 'flashlight_enabled',
                'nfc_enabled', 'unknown_sources', 'developer_mode']

        df = load_df('samples.parquet', cols)

        df.drop_duplicates(inplace=True)


        df.insert(0, 'services_id', np.packbits(df.values, axis=-1))
        df.sort_values(by=['services_id'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        save_df(df, 'services_dimension.parquet')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
