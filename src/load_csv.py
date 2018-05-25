#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from utils import load_dtypes


def main():
    try:
        if len(sys.argv) < 2:
            raise IOError('Dataset missing!')

        dname = sys.argv[1].split('.')[0] + '.pkl'

        # cannot parse dates!
        gl = pd.read_csv(sys.argv[1], dtype=load_dtypes(dname))
        gl.info(memory_usage='deep')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
