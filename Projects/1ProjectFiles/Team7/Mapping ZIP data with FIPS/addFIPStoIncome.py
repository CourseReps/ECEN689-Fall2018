import pandas as pd
import numpy as np

def map_fips(input_filename_1, input_filename_2, output_filename):

    eight_df = pd.read_csv(input_filename_1)

    zip_county = pd.read_csv(input_filename_2)

    zip_county_extract = zip_county[['ZIP', 'STCOUNTYFP']]

    zip_dict = zip_county_extract.set_index('ZIP')['STCOUNTYFP'].to_dict()

    eight_extract = eight_df[['state', 'ZIPCODE', 'agi_class', 'n1', 'a00100']]
    eight_zips = eight_extract['ZIPCODE'].tolist()

    zip_fips1 = []

    # zip_dict[99629] = 2170
    # zip_dict[99501] = 2170
    # zip_dict[99710] = 2090
    # zip_dict[99711] = 2090
    # zip_dict[99802] = 2110
    # zip_dict[99803] = 2110
    # zip_dict[99821] = 2110
    # zip_dict[35011] = 12011
    # zip_dict[35161] = 1121
    # zip_dict[35662] = 1033
    # zip_dict[36304] = 1069
    # zip_dict[36331] = 1031
    # zip_dict[36427] = 1251
    #print(zip_dict[99629])
    missing_zips = []
    missing_indices = []
    j = 0
    for i in eight_zips:
        #print(j)
        try:
            zip_fips1.append(zip_dict[int(i)])
            j = j+1
        except(KeyError):
            missing_zips.append(i)
            missing_indices.append(j)
            j = j+1

    for i in missing_indices:
        eight_extract = eight_extract.drop(index=i, axis=0)

    eight_extract['FIPS'] = zip_fips1
    eight_extract.to_csv(output_filename, index=False)

