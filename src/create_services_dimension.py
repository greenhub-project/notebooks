import sys
import numpy as np
import pandas as pd
from utils import load_df, save_df

def main():
    try:
        #if path isn't given (csv files are in the same directory as script)
        if len(sys.argv) == 1:
            cols = ['bluetooth_enabled', 'location_enabled', 'power_saver_enabled', 'flashlight_enabled', 'nfc_enabled', 'unknown_sources', 'developer_mode']
            df = pd.read_csv('settings.csv', usecols=cols)

        #if path is given (path to the csv files folder)
        elif len(sys.argv) == 2:
            cols = ['bluetooth_enabled', 'location_enabled', 'power_saver_enabled', 'flashlight_enabled', 'nfc_enabled', 'unknown_sources', 'developer_mode']
            df = pd.read_csv(sys.argv[1] + '\settings.csv', usecols=cols)

        else:
            raise IOError('Dataset missing!')

        df.drop_duplicates(inplace=True)

        df.insert(0, 'services_id', np.packbits(df.values, axis=-1))
        df.sort_values(by=['services_id'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        save_df(df, 'services_dimension.parquet')

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
