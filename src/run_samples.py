#!/usr/bin/env python3

import numpy as np
import pandas as pd


def main():
    try:
        gl = pd.read_parquet('samples.parquet.gzip')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
