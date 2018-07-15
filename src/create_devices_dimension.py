import sys
import numpy as np
import pandas as pd
from utils import save_df, downcastDfTypes


def createDevicesDimension(Dataset_Path):
    cols = ['id', 'model', 'manufacturer', 'brand', 'product', 'os_version', 'kernel_version', 'is_root']
    devices_df = pd.read_csv(Dataset_Path + '\devices.csv', usecols=cols)

    devices_df = devices_df.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

    devices_df = downcastDfTypes(devices_df)

    save_df(devices_df, 'devices_dimension.parquet')
