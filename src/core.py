#!/usr/bin/env python3

from utils import load_df

def main():
    try:
        gl = load_df('samples.parquet')
        gl.info(memory_usage='deep')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
