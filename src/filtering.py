import numpy as np
import pandas as pd
from utils import load_df, pack_comb, unpack_comb


def main():
    try:
        df = load_df('combinations.parquet')

        # ['screen_on', 'bluetooth_enabled', 'location_enabled', 'power_saver_enabled',
        #  'nfc_enabled', 'unknown_sources', 'developer_mode', 'wifi_enabled']

        # filtering
        haystack = pack_comb([0, 1, 0, 0, 0, 0, 0, 0])

        

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
