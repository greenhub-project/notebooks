import sys
import numpy as np
import pandas as pd
from itertools import islice


def compare(previous, current, following):
    if previous < 0 and current > 0 and following > 0:
        return True
    if previous > 0 and current < 0 and following < 0:
        return True
    else:
        return False


def createFactsTable(Dataset_Path):
    # ------------------------------ recives dataset folder path and load csv files to one big joined df ------------------------------
    cols = ['sample_id', 'charger']
    df_battery = pd.read_csv(Dataset_Path + '\\battery_details.csv', usecols=cols)

    cols = ['id', 'device_id', 'timestamp', 'battery_state', 'battery_level', 'screen_on']
    df_samples = pd.read_csv(Dataset_Path + '\samples.csv', usecols=cols, parse_dates=['timestamp'])

    cols = ['sample_id', 'bluetooth_enabled', 'location_enabled', 'power_saver_enabled', 'flashlight_enabled', 'nfc_enabled', 'developer_mode']
    df_settings = pd.read_csv(Dataset_Path + '\settings.csv', usecols=cols)

    df_samples.rename(columns={"id": "sample_id"}, inplace=True)

    merged_df = df_samples.merge(df_battery, on='sample_id', how='inner')
    merged_df = merged_df.merge(df_settings, on='sample_id', how='inner')
    merged_df = merged_df[merged_df.timestamp >= pd.Timestamp('2017-10-15')]

    merged_df.drop('sample_id', axis=1, inplace=True)

    print('Samples df created...')
    # ------------------------------ cleaning data and pre computing new information (delta_battery_level, delta_time) ------------------------------
    sorted_samples = merged_df.sort_values(by=['device_id', 'timestamp']).reset_index(drop=True)
    sorted_samples['delta_battery_level'] = sorted_samples.battery_level.diff()
    sorted_samples['delta_time'] = round((sorted_samples.timestamp.diff().dt.seconds) / 3600, 2)
    sorted_samples.loc[sorted_samples.device_id != sorted_samples.device_id.shift(), 'delta_battery_level'] = None
    sorted_samples = sorted_samples.drop(sorted_samples[sorted_samples.delta_battery_level == 0].index)
    sorted_samples.reset_index(drop=True, inplace=True)
    print('Samples df cleanned...')

    # ------------------------------  pre computing new information (period) ------------------------------
    sorted_samples['period'] = pd.NaT
    sorted_samples.at[0, 'period'] = sorted_samples.at[0, 'timestamp']

    for ind in islice(sorted_samples.index, 1, len(sorted_samples)-2):
        if compare(sorted_samples.at[ind-1, 'delta_battery_level'], sorted_samples.at[ind, 'delta_battery_level'], sorted_samples.at[ind+1, 'delta_battery_level']):
            sorted_samples.at[ind-1, 'period'] = sorted_samples.at[ind-1, 'timestamp']
            sorted_samples.at[ind, 'period'] = sorted_samples.at[ind, 'timestamp']

            sorted_samples.at[ind-1, 'delta_time'] = None
            sorted_samples.at[ind, 'delta_time'] = None

    sorted_samples.loc[sorted_samples.device_id != sorted_samples.device_id.shift(), 'period'] = sorted_samples.timestamp
    sorted_samples.loc[sorted_samples.device_id != sorted_samples.device_id.shift(), 'delta_time'] = None
    print('Samples df pre calculations finished...')

    # ------------------------------  computing facts table ------------------------------
    facts_table = pd.DataFrame(columns=['device_id', 'time_diff', 'battery_diff', 'rate', 'charging', 'number_samples',
                                        'screen_on_time', 'screen_off_time', 'bluetooth_enabled_time', 'location_enabled_time',
                                        'power_saver_enabled_time', 'flashlight_enabled_time', 'nfc_enabled_time',
                                        'developer_mode_time'])

    period_times = sorted_samples[sorted_samples.period.notnull()].period

    p_index = period_times.index
    for i, value in enumerate(period_times):
        if i%2 != 0:
            period_final_index = p_index[i]
            period_first_index = p_index[i-1]
            period_final_date = value
            period_first_date = period_times.iloc[i-1]

            period = sorted_samples[period_first_index:period_final_index + 1]
            diff_time = (period_final_date - period_first_date).seconds / 3600

            if diff_time == 0:
                continue

            diff_battery = sorted_samples.at[period_final_index, 'battery_level'] - sorted_samples.at[period_first_index, 'battery_level']
            rate = diff_battery / diff_time

            number_samples = (period_final_index - period_first_index) + 1
            screen_on_time = period[period.screen_on == 1].delta_time.sum()
            screen_off_time = diff_time - screen_on_time

            bluetooth_enabled_time = period[period.bluetooth_enabled == 1].delta_time.sum()
            location_enabled_time = period[period.location_enabled == 1].delta_time.sum()
            power_saver_enabled_time = period[period.power_saver_enabled == 1].delta_time.sum()
            flashlight_enabled_time = period[period.flashlight_enabled == 1].delta_time.sum()
            nfc_enabled_time = period[period.nfc_enabled == 1].delta_time.sum()
            developer_mode_time = period[period.developer_mode == 1].delta_time.sum()

            diff_time = round(diff_time, 2)
            diff_battery = round(diff_battery, 2)
            rate = round(rate, 3)
            screen_on_time = round(screen_on_time, 2)
            screen_off_time = round(screen_off_time, 2)
            bluetooth_enabled_time = round(bluetooth_enabled_time, 2)
            location_enabled_time = round(location_enabled_time, 2)
            power_saver_enabled_time = round(power_saver_enabled_time, 2)
            flashlight_enabled_time = round(flashlight_enabled_time, 2)
            nfc_enabled_time = round(nfc_enabled_time, 2)
            developer_mode_time = round(developer_mode_time, 2)

            if rate < 0:
                facts_table.loc[-1] = [sorted_samples.at[period_first_index, 'device_id'],
                                       diff_time,
                                       diff_battery,
                                       rate,
                                       False,
                                       number_samples,
                                       screen_on_time,
                                       screen_off_time,
                                       bluetooth_enabled_time,
                                       location_enabled_time,
                                       power_saver_enabled_time,
                                       flashlight_enabled_time,
                                       nfc_enabled_time,
                                       developer_mode_time]
                facts_table.index = facts_table.index + 1
            else:
                facts_table.loc[-1] = [sorted_samples.at[period_first_index, 'device_id'],
                                       diff_time,
                                       diff_battery,
                                       rate,
                                       True,
                                       number_samples,
                                       screen_on_time,
                                       screen_off_time,
                                       bluetooth_enabled_time,
                                       location_enabled_time,
                                       power_saver_enabled_time,
                                       flashlight_enabled_time,
                                       nfc_enabled_time,
                                       developer_mode_time]
                facts_table.index = facts_table.index + 1

    facts_table.to_csv('facts_table.csv', encoding='utf-8', index=False)
    print('Facts Table Created!')

def main():
    try:
        if len(sys.argv) == 2:
            print('Creating Facts Table ...')
            createFactsTable(sys.argv[1])
            print('Done!!!\n')

        else:
            raise IOError('Dataset missing!')

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
