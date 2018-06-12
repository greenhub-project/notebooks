import sys
import numpy as np
import pandas as pd
from utils import load_df, save_df

def main():
    try:
        if len(sys.argv) < 2:
            raise IOError('Dataset missing!')

        cols = ['timestamp']

        df = load_df('samples.parquet', cols)

        df['day'] = df.timestamp.dt.day
        df['month'] = df.timestamp.dt.month
        df['year'] = df.timestamp.dt.year

        df.drop('timestamp', axis=1, inplace=True)
        df.drop_duplicates(inplace=True)
        df.sort_values(by=['year', 'month', 'day'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.insert(0, 'time_id', np.arange(1, len(df) + 1))

        save_df(df, 'time_dimension.parquet')

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
