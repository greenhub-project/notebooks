import numpy as np
import pandas as pd
from itertools import islice
from utils import load_df, save_df, downcastDfTypes


def computeSubsetMean(subset, first_element_index):
    subset_state = subset.at[first_element_index, 'battery_state']

    if subset_state == 'charging':
        charge_rows = subset[subset['time_diff'] > 0]

        if len(charge_rows.index) == 0:
            return -1, -1
        else:
            return np.nanmean(charge_rows['time_diff']), np.nanstd(charge_rows['time_diff'])

    elif subset_state == 'discharging':
        discharge_rows = subset[subset['time_diff'] < 0]

        if len(discharge_rows.index) == 0:
            return -1, -1
        else:
            return np.nanmean(discharge_rows['time_diff']), np.nanstd(discharge_rows['time_diff'])

    else:
        charge_rows = subset[subset['time_diff'] > 0]
        discharge_rows = subset[subset['time_diff'] < 0]

        if (len(charge_rows) > 0) and (len(charge_rows) > round(len(subset) * 0.9)):
            return np.nanmean(charge_rows['time_diff']), np.nanstd(charge_rows['time_diff'])

        elif (len(discharge_rows) > 0) and (len(discharge_rows) > round(len(subset) * 0.9)):
            return np.nanmean(discharge_rows['time_diff']), np.nanstd(discharge_rows['time_diff'])

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
            subset = samples_df[previous_index:ind].copy()

            if len(subset) >= 10:
                subset.loc[abs(subset['time_diff']) > 3600, 'time_diff'] = None
                mean, std = computeSubsetMean(subset, previous_index)

                if (mean != -1) and not np.isnan(mean):
                    buildFactsTable(subset, facts_table, previous_index, mean, std)

            previous_index = ind

    #final samples
    subset = samples_df[previous_index:len(samples_df)].copy()

    if len(subset) >= 10:
        subset.loc[abs(subset['time_diff']) > 3600, 'time_diff'] = None

        mean, std = computeSubsetMean(subset, previous_index)

        if (mean != -1) and not np.isnan(mean):
            buildFactsTable(subset, facts_table, previous_index, mean, std)


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
