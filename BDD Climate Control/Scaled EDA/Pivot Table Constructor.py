# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:01:52 2020

@author: ACHOPRA8
"""
csv_file = r'C:\Users\achopra8\Downloads\2019f150VIN1FTEW1EP1KFA04560.csv'
VIN = csv_file[csv_file.index('VIN') + 3:csv_file.index('.')]

import pandas as pd
from sklearn.impute import KNNImputer

signals = ['AirAmb_Te_ActlFilt',
 'Outside_Air_Temp_Stat',
 'HvacEvap_Te_Actl',
 'HvacEvap_Te_Rq',
 'AirCondFluidHi_P_Actl',
 'CoolantFanStepAct',
 'VehLong2_A_Actl',
 'HvacAirCond_B_Rq',
 'Veh_V_ActlEng',
 'ApedPos_Pc_ActlArb',
 'EngAout_N_Actl',
 'HvacBlwrFront_D_Stat',
 'BattULo_U_Actl',
 'GearLvrPos_D_Actl',
 'Ignition_Status',
 'BrkTot_Tq_Actl',
 'OdometerMasterValue']
dtypes = [str if i in [5, 7, 11, 13] else float for i in range(len(signals))]
agg_funcs = ['mean' if dtype == float else pd.Series.mode for dtype in dtypes] # functions to aggregate each signal by

df = pd.read_csv(csv_file, nrows=20000000)
df['cvdcus_timestamp_s_3'] = df['cvdcus_timestamp_s_3'].astype('datetime64[ns]') # ensure timestamps are in datetime format
df['cvdcus_timestamp_s_3'] = df['cvdcus_timestamp_s_3'].dt.round('1s') # round to nearest second

# helper function to make pivot table; allows to agg by different functions
def df_by_sig(df, signals=signals, dtypes=dtypes, agg_funcs=agg_funcs):
    grouped_by_sig = []
    invalid_sig = []
    invalid_val = []
    for i in range(len(signals)):
        try:
            temp = df[df['cvdcus_dcd_sig_n_x_3'] == signals[i]]
            temp['cvdcus_dcd_sig_val_str_x_3'] = temp['cvdcus_dcd_sig_val_str_x_3'].astype(dtypes[i])
            temp = temp.groupby(['cvdcus_timestamp_s_3', 'cvdcus_dcd_sig_n_x_3']).agg(agg_funcs[i]).reset_index()
            temp = temp.rename(columns={'cvdcus_dcd_sig_val_str_x_3':temp['cvdcus_dcd_sig_n_x_3'].iloc[0]}).drop('cvdcus_dcd_sig_n_x_3', axis=1)
            grouped_by_sig.append(temp)
        except ValueError as err:
            err = str(err)
            invalid_sig.append(signals[i])
            invalid_val.append(err[err.index("'") + 1:-1])
    return grouped_by_sig, invalid_sig, invalid_val

dfs_grouped_by_sig, invalid_sig, invalid_val = df_by_sig(df)
if invalid_sig:
    df = df.drop(df[(df['cvdcus_dcd_sig_n_x_3'].isin(invalid_sig)) & (df['cvdcus_dcd_sig_val_str_x_3'].isin(invalid_val))].index).reset_index(drop=True)
    dfs_grouped_by_sig = df_by_sig(df)[0]

# construct pivot table
pivotted = pd.concat([df.set_index('cvdcus_timestamp_s_3') for df in dfs_grouped_by_sig], axis=1).reset_index()
pivotted['epoch'] = pivotted['cvdcus_timestamp_s_3'].astype('int64')/1e9
cols = pivotted.columns.to_list()
pivotted = pivotted[[cols[0]] + [cols[-1]] + cols[1:-1]]

for signal, dtype in zip(signals, dtypes):
    if signal == 'Ignition_Status':
        continue
    pivotted[signal] = pivotted[signal].astype(dtype)
pivotted = pivotted.replace({r"\['(\w+)'.*": r'\1'}, regex=True) # choose one mode when > 1 present
pivotted = pivotted.replace({'nan':float('nan')})

# calc epoch diff to define trip
pivotted['epoch_diff'] = pivotted['epoch'].diff() # add col to show diff between epoch
cols = list(pivotted.columns)
pivotted = pivotted[cols[0:2] + [cols[-1]] + cols[2:-1]]

# make empty col that will flag trips
pivotted['trip'] = float('nan')
cols = list(pivotted.columns)
pivotted = pivotted[cols[0:3] + [cols[-1]] + cols[3:-1]]

def flag_trips(pivotted):
    # add flags to indicate trips
    trip_flag = 1
    for i in [0] + list(pivotted[pivotted['epoch_diff'] > 1800].index):
        pivotted.loc[i, 'trip'] = trip_flag
        trip_flag += 1
    pivotted['trip'] = pivotted['trip'].fillna(method='ffill')
    pivotted['trip'] = pivotted['trip'].astype(int)

flag_trips(pivotted)

# filter trips
def filter_trips(pivotted):
    trips_to_filter_out = pivotted.groupby('trip')['epoch'].agg(lambda x: x.max() - x.min())
    trips_to_filter_out = trips_to_filter_out[(trips_to_filter_out < 60) | (trips_to_filter_out > 43200)].index # keep trips w/ duration > 60 even if odo change is <= 0
    pivotted = pivotted[~pivotted['trip'].isin(trips_to_filter_out)].reset_index(drop=True) # filter out
    pivotted['epoch_diff'] = pivotted['epoch'].diff() # recalc epoch diff since some trips now 
    pivotted['trip'] = pivotted['trip'].replace({trip: i + 1 for i, trip in enumerate(pivotted['trip'].unique())}) # reset trip flags
    return pivotted

pivotted = filter_trips(pivotted)

pivotted['temp_gap'] = pivotted['HvacEvap_Te_Actl'] - pivotted['HvacEvap_Te_Rq'] # add temp gap col

# impute missing values
def make_dummies_and_impute(trip_num):
    cols_to_ffill = ['BrkTot_Tq_Actl', 'Veh_V_ActlEng', 'GearLvrPos_D_Actl','ApedPos_Pc_ActlArb', 'EngAout_N_Actl', 'HvacEvap_Te_Actl', 'HvacEvap_Te_Rq', 'HvacAirCond_B_Rq', 'BattULo_U_Actl', 'AirAmb_Te_ActlFilt', 'Outside_Air_Temp_Stat', 'OdometerMasterValue', 'VehLong2_A_Actl', 'temp_gap']
    trip = pivotted[pivotted['trip'] == trip_num]
    trip = trip[['epoch'] + list(trip.columns[4:])]
    trip[cols_to_ffill] = trip[cols_to_ffill].fillna(method='ffill')
    imputer = KNNImputer(weights='distance')
    trip_one_hot_encoded = pd.get_dummies(trip)
    trip = pd.DataFrame(imputer.fit_transform(trip_one_hot_encoded), index=trip.index, columns=trip_one_hot_encoded.columns)
    return trip_one_hot_encoded

pivotted_f = []
trips_to_filter_out = []
for trip_num in pivotted['trip'].unique():
    try:
        pivotted_f.append(make_dummies_and_impute(trip_num))
    except ValueError:
        trips_to_filter_out.append(trip_num)

if trips_to_filter_out:
    pivotted = pivotted[~pivotted['trip'].isin(trips_to_filter_out)].reset_index(drop=True) # filter out
    pivotted['epoch_diff'] = pivotted['epoch'].diff() # recalc epoch diff since some trips now 
    pivotted['trip'] = pivotted['trip'].replace({trip: i + 1 for i, trip in enumerate(pivotted['trip'].unique())}) # reset trip flags
    pivotted_f = [make_dummies_and_impute(trip_num) for trip_num in pivotted['trip'].unique()]
    
pivotted_f = pd.concat(pivotted_f)
pivotted_f = pd.concat([pivotted[pivotted.columns[:4]], pivotted_f], axis=1)
pivotted_f = pivotted_f.fillna(method='ffill').fillna(method='bfill') # remaining nans

# reverse one-hot encoding
categorical_vars = ['HvacBlwrFront_D_Stat', 'GearLvrPos_D_Actl', 'HvacAirCond_B_Rq', 'CoolantFanStepAct']
for var in categorical_vars:
    all_cols = pivotted_f.columns[17:]
    rel_cols = [col for col in all_cols if col.startswith(var) ]
    pivotted_f[var] = pivotted_f[rel_cols].idxmax(axis=1)

for var in categorical_vars:
    pivotted_f[var] = pivotted_f[var].map(lambda x: x[(x.index(var) + len(var) + 1):])

pivotted_f = pivotted_f[list(pivotted_f.columns[:17]) + list(pivotted_f.columns[-4:])]

pivotted_f.to_csv(r'C:\Users\achopra8\Documents\BDD Climate Control\Pivotted Data\{}.csv'.format(VIN))

