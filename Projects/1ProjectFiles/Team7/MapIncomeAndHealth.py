import pandas as pd
import numpy as np

def map_function(input_filename_1, input_filename_2, output_filename):

    taken_df = pd.read_csv(input_filename_1)
    second_df = pd.read_csv(input_filename_2)

    #print(second_df.dtypes.index)

    if ('08' in input_filename_1):
        dia_df = second_df[['FIPS', 'PCT_DIABETES_ADULTS08']]
        dia_dict = dia_df.set_index('FIPS')['PCT_DIABETES_ADULTS08'].to_dict()

        obs_df = second_df[['FIPS', 'PCT_OBESE_ADULTS08']]
        obs_dict = obs_df.set_index('FIPS')['PCT_OBESE_ADULTS08'].to_dict()

    if ('13' in input_filename_1):
        dia_df = second_df[['FIPS', 'PCT_DIABETES_ADULTS08']]
        dia_dict = dia_df.set_index('FIPS')['PCT_DIABETES_ADULTS13'].to_dict()

        obs_df = second_df[['FIPS', 'PCT_OBESE_ADULTS08']]
        obs_dict = obs_df.set_index('FIPS')['PCT_OBESE_ADULTS13'].to_dict()

    FIPS_income = taken_df['FIPS'].values.tolist()

    mapped_dia = []
    mapped_obs = []

    i = 0
    missing_fips =[]
    missing_indices = []
    for FIPS in FIPS_income:
        try:
            mapped_dia.append(dia_dict[FIPS])
            mapped_obs.append(obs_dict[FIPS])
            print(i)
            i += 1
        except(KeyError):
            missing_fips.append(FIPS)
            missing_indices.append(i)
            print(i)
            i += 1
    print(len(missing_indices))

    for i in missing_indices:
        taken_df = taken_df.drop(index=i, axis=0)

    taken_df['DIABETES_ADULTS'] = mapped_dia
    taken_df['OBESE_ADULTS'] = mapped_obs

    taken_df.to_csv(output_filename,index=False)
