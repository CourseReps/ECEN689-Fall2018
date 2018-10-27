# ECEN 689-606
# Project 1
# Team 8
# Mason Rumuly
# Format related portions of original data for easy accessibility

#########################################################################
# imports

from inspect import getsourcefile
from os import path

import sys

import numpy as np
import pandas as pd

#########################################################################
# paths and file names

# absolute path to home folder
home_dir = path.sep.join(path.abspath(getsourcefile(lambda:0)).split(path.sep)[:-1]) + path.sep

# extracted data file
health_med_income_fn = 'fips_health_med_income.csv'

#########################################################################
# functions to retrieve data dataframe

def get_data():
    '''
    Returns pandas dataframe containing food atlas health indicators
    '''
    return pd.read_csv(
        home_dir + health_med_income_fn,
        index_col=0
    )

#########################################################################
# main, use for extracting original data
if __name__ == '__main__':

    # absolute path to download directory of raw data
    source_dir = path.sep.join(home_dir.split(path.sep)[:3]) + '{0}Downloads{0}'.format(path.sep)

    # original source file names
    atlas_fn = 'DataDownload.xls'
    irs08_fn = '08zpall.csv'
    irs13_fn = 'zipcodeagi13.csv'
    map_fn = 'ZIP-COUNTY-FIPS.csv'

    # grab health data from food atlas
    health_df = pd.read_excel(
        source_dir + atlas_fn, 
        sheet_name='HEALTH',
        header=0,
        index_col=0,
        usecols='A,D:G'
    ).dropna()
    # grab median income data from food atlas
    med_income_df = pd.read_excel(
        source_dir + atlas_fn,
        sheet_name='SOCIOECONOMIC',
        header=0,
        index_col=0,
        usecols='A,L'
    ).dropna()

    # combine health data and median income
    health_med_df = health_df.join(med_income_df, how='inner')

    # grab income data from IRS statistics
    income08_df = pd.read_csv(
        source_dir + irs08_fn,
        header=0,
        names=['zipcode', 'agi_bracket', 'total_returns', 'joint_returns', 'exemptions', 'dependents', 'agi_kUSD'],
        index_col=None,
        usecols=[1,2,3,4,6,7,8],
        dtype={'zipcode': str}
    ).dropna().groupby('zipcode').agg(sum)
    income08_df['agi_kUSD'] = income08_df['agi_kUSD'] / 1000
    income13_df = pd.read_csv(
        source_dir + irs13_fn,
        header=0,
        names=['zipcode', 'agi_bracket', 'total_returns', 'joint_returns', 'exemptions', 'dependents', 'agi_kUSD'],
        index_col=None,
        usecols=[2,3,4,6,9,10,11],
        dtype={'zipcode':object}
    ).dropna().groupby('zipcode').agg(sum) 
    
    # combine income data years
    income_df = income08_df.join(income13_df, how='inner', lsuffix='_08', rsuffix='_13')

    # get ZIP to FIPS map
    map_series = pd.read_csv(
        home_dir + map_fn,
        header=0,
        names=['zipcode', 'FIPS'],
        index_col=None,
        usecols=[0,3],
        squeeze=False,
        dtype={'zipcode':object}
    )
    map_series = map_series.drop('FIPS', axis=1).set_index(map_series['FIPS'])

    # summate income stats into FIPS areas
    health_med_income_df = health_med_df
    for c in income_df.columns.values:
        health_med_income_df[c] = np.nan

        for fips in health_med_df.index.values:
            if not fips in map_series.index:
                break
            
            for zips in map_series.loc[fips,'zipcode']:
                if not zips in income_df.index:
                    break

                if np.isnan(health_med_income_df.loc[fips, c]):
                    health_med_income_df.loc[fips, c] = 0

                health_med_income_df.loc[fips, c] += income_df.loc[zips, c]
    health_med_income_df = health_med_income_df.dropna()

    # save extracted data
    health_med_income_df.to_csv(home_dir + health_med_income_fn)
