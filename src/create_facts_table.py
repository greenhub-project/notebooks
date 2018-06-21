import numpy as np
import pandas as pd
from itertools import islice
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


def computeSubsetMean(subset, first_element_index):
    subset_state = subset.at[first_element_index, 'battery_state']

    if subset_state == 'charging':
        if len(subset[subset['time_diff'] > 0]) >= round(len(subset) * 0.9):
            return np.nanmean(subset['time_diff']), np.nanstd(subset['time_diff'])
        else:
            return -1, -1
    elif subset_state == 'discharging':
        if len(subset[subset['time_diff'] < 0]) >= round(len(subset) * 0.9):
            return np.nanmean(subset['time_diff']), np.nanstd(subset['time_diff'])
        else:
            return -1, -1
    else:
        if (len(subset[subset['time_diff'] < 0]) >= round(len(subset) * 0.9)) or (len(subset[subset['time_diff'] > 0]) >= round(len(subset) * 0.9)):
            return np.nanmean(subset['time_diff']), np.nanstd(subset['time_diff'])
        else:
            return -1, -1


def buildFactsTable(subset, facts_table, previous_index, mean, std):
    if mean > 0:
        if 'full' in subset['battery_state'].values:
            facts_table.loc[-1] = [subset.at[previous_index, 'device_id'],
                                   subset.at[previous_index, 'time_id'],
                                   subset.at[previous_index, 'service_comb_id'], -1, mean, 1, std]
            facts_table.index = facts_table.index + 1

        else:
            facts_table.loc[-1] = [subset.at[previous_index, 'device_id'],
                                   subset.at[previous_index, 'time_id'],
                                   subset.at[previous_index, 'service_comb_id'], -1,  mean, 0, std]
            facts_table.index = facts_table.index + 1
    else:
        facts_table.loc[-1] = [subset.at[previous_index, 'device_id'],
                               subset.at[previous_index, 'time_id'],
                               subset.at[previous_index, 'service_comb_id'], abs(mean), -1, -1, std]
        facts_table.index = facts_table.index + 1


def computeSubsets(samples_df, facts_table):
    previous_index = 1

    for ind, v in islice(samples_df['time_diff'].iteritems(), 1, None):
        if np.isnan(v):
            subset = samples_df[previous_index:ind]

            if len(subset) >= 10:
                mean, std = computeSubsetMean(subset, previous_index)

                if mean != -1:
                    buildFactsTable(subset, facts_table, previous_index, mean, std)

            previous_index = ind


def main():
    cols= ['device_id', 'service_comb_id', 'time_id', 'timestamp', 'battery_state', 'battery_level', 'time_diff']
    samples_df = load_df('processed_samples.parquet', cols)
    facts_table = pd.DataFrame(columns=['device_id','time_id','services_id','discharge_per_unit','charge_per_unit','reach_full', 'standard_dev'])

    print('Computing facts table, may take a while...')
    computeSubsets(samples_df, facts_table)

    facts_table.device_id = facts_table.device_id.astype(int)
    facts_table.time_id = facts_table.time_id.astype(int)
    facts_table.services_id = facts_table.services_id.astype(int)
    facts_table.discharge_per_unit = facts_table.discharge_per_unit.apply(lambda x: round(x, 2))
    facts_table.charge_per_unit = facts_table.charge_per_unit.apply(lambda x: round(x, 2))
    facts_table.reach_full = facts_table.reach_full.astype(int)
    facts_table.standard_dev = facts_table.standard_dev.apply(lambda x: round(x, 2))

    facts_table = downcastDfTypes(facts_table)
    facts_table.info(memory_usage='deep')

    save_df(facts_table, 'facts_table.parquet')


if __name__ == '__main__':
    main()
