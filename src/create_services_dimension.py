import sys
import numpy as np
import pandas as pd
from utils import save_df, downcastDfTypes


def CreateServicesDimension(Dataset_Path):
    cols = ['bluetooth_enabled', 'location_enabled', 'power_saver_enabled', 'flashlight_enabled', 'nfc_enabled', 'unknown_sources', 'developer_mode']
    df = pd.read_csv(Dataset_Path + '\settings.csv', usecols=cols)

    df.drop_duplicates(inplace=True)

    df.insert(0, 'services_id', np.packbits(df.values, axis=-1))
    df.sort_values(by=['services_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = downcastDfTypes(df)

    save_df(df, 'services_dimension.parquet')
