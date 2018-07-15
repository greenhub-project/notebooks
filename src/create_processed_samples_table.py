import sys
import numpy as np
import pandas as pd
from utils import save_df, downcastDfTypes


def createProcessSamplesTable(Dataset_Path):
    # ------------------------------ recives dataset folder path and load csv files to one big joined df ------------------------------
    cols = ['id', 'device_id', 'timestamp', 'battery_state', 'battery_level', 'screen_on']
    df_samples = pd.read_csv(Dataset_Path + '\samples.csv', usecols=cols, parse_dates=['timestamp'])

    cols = ['sample_id', 'bluetooth_enabled', 'location_enabled', 'power_saver_enabled', 'flashlight_enabled', 'nfc_enabled', 'unknown_sources', 'developer_mode']
    df_settings = pd.read_csv(Dataset_Path + '\settings.csv', usecols=cols)

    df_samples.set_index('id', inplace=True)
    df_settings.set_index('sample_id', inplace=True)

    joined_samples = df_samples.join(df_settings)

    print('------>Before:<------')
    joined_samples.info(memory_usage='deep')

    # ------------------------------ cleaning data and pre computing new information (time_diff, ) ------------------------------

    # sorting
    joined_samples = joined_samples.sort_values(by=['device_id', 'timestamp'])

    # filtering
    joined_samples = joined_samples[joined_samples.timestamp >= pd.Timestamp('2017-10-15')]

    # remove duplicate samples
    joined_samples.drop_duplicates(subset=['timestamp', 'battery_state', 'battery_level'], inplace=True)

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

    # ------------------------------ downcast all types datfarame and crete binary file -----------------------------------
    # special cast battery level
    joined_samples['battery_level'] = joined_samples['battery_level'].astype(np.uint8)

    joined_samples = downcastDfTypes(joined_samples)

    print('------------------------------')
    print('------>After:<------')
    joined_samples.info(memory_usage='deep')
    print('------------------------------')

    save_df(joined_samples, 'processed_samples.parquet')
