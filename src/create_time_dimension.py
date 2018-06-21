import sys
import numpy as np
import pandas as pd
from utils import load_df, mem_usage, save_df, typecast_objects, typecast_ints, typecast_floats

def downcastDfTypes(df):
    # downcast integer columns
    converted_int = typecast_ints(df.select_dtypes(include=['integer']))

    # downcast float columns
    converted_float = typecast_floats(df.select_dtypes(include=['float']))

    # convert object columns to lowercase
    df_obj = df.select_dtypes(include=['object'])
    df_obj = df_obj.apply(lambda x: x.str.lower())

    # convert object to category columns
    # when unique values < 50% of total
    converted_obj = typecast_objects(df_obj)

    # transform optimized types
    df[converted_int.columns] = converted_int
    df[converted_float.columns] = converted_float
    df[converted_obj.columns] = converted_obj

    return df

def main():
    samples_df = load_df('processed_samples.parquet', None)

    samples_df['day'] = samples_df.timestamp.dt.day
    samples_df['month'] = samples_df.timestamp.dt.month
    samples_df['year'] = samples_df.timestamp.dt.year

    time_df = samples_df[['day', 'month', 'year']].copy()
    time_df.drop_duplicates(inplace=True)
    time_df.sort_values(by=['year', 'month', 'day'], inplace=True)
    time_df.reset_index(drop=True, inplace=True)
    time_df.insert(0, 'time_id', np.arange(1, len(time_df) + 1))

    merge_df = pd.merge(samples_df, time_df, how='left', left_on=['day', 'year', 'month'], right_on=['day', 'year', 'month'])
    merge_df.drop(['day', 'month', 'year'], axis=1, inplace=True)
    merge_df = merge_df[['device_id', 'service_comb_id', 'time_id', 'timestamp', 'battery_state', 'battery_level', 'time_diff']]

    time_df = downcastDfTypes(time_df)
    merge_df = downcastDfTypes(merge_df)

    save_df(time_df, 'time_dimension.parquet')
    save_df(merge_df, 'processed_samples.parquet')


if __name__ == '__main__':
    main()
