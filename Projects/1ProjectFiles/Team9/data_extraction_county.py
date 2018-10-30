
# This code is used for extracting county-wise data from supplemental_data, health data and food security data
# This does a left-join for ensuring FIPS consistency

import pandas as pd
import numpy as np

COUNTY_DTYPES = {'FIPS': str, 'State': str, 'County': str}
df1 = pd.read_csv('datasets/Food_Atlas_supplemental_data.csv', encoding='cp1252', dtype=COUNTY_DTYPES)
df2 = pd.read_csv('datasets/Food_Atlas_health_data.csv', encoding='cp1252', dtype=COUNTY_DTYPES)
df3 = pd.read_csv('datasets/Food_Atlas_food_security.csv', encoding='cp1252', dtype=COUNTY_DTYPES)

# left join for supplemental data, health data
df_intermed = df1.merge(df2, left_on='FIPS', right_on='FIPS', how='left')


# left join for previous data, food security
df_final = df_intermed.merge(df3, left_on='FIPS', right_on='FIPS', how='left')

# final columns to extract
col_list = ['FIPS', 'State', 'County', 'Population Estimate, 2013', 'PCT_DIABETES_ADULTS13', 'PCT_OBESE_ADULTS13', 'FOODINSEC_13_15', 'VLFOODSEC_13_15']
df = df_final[col_list]

# save to csv
df.to_csv('Food_Atlas_County_2013.csv', index=False)

