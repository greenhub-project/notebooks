import sys
from create_processed_samples_table import createProcessSamplesTable
from create_services_dimension import CreateServicesDimension
from create_devices_dimension import createDevicesDimension
from create_time_dimension import createTimeDimension

def main():
    try:
        if len(sys.argv) == 2:
            print('Creating Processed Samples Parquet File...')
            createProcessSamplesTable(sys.argv[1])
            print('Done!!!\n')

            print('Creating Services Dimension Parquet File...')
            CreateServicesDimension(sys.argv[1])
            print('Done!!!\n')

            print('Creating Devices Dimension Parquet File...')
            createDevicesDimension(sys.argv[1])
            print('Done!!!\n')

            print('Creating Time Dimension Parquet File...')
            createTimeDimension()
            print('Done!!!\n')


        else:
            raise IOError('Dataset missing!')


    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
