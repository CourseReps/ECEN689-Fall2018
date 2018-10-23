# ECEN 689-606
# Project 1
# Team 8
# Mason Rumuly
# Format related portions of original data for easy accessibility

#########################################################################
# imports

from inspect import getsourcefile
from os import path

import pandas as pd

#########################################################################
# paths and file names

# absolute path to home folder
home_dir = path.sep.join(path.abspath(getsourcefile(lambda:0)).split(path.sep)[:-1]) + path.sep

# food atlas extracted file
health_fn = 'fips_health.csv'

# income extracted files
income_fn = 'zip_income.csv'

#########################################################################
# functions to retrieve data as dataframes

def get_health():
    '''
    Returns pandas dataframe containing food atlas health indicators
    '''
    return pd.read_csv(
        home_dir + health_fn,
        index_col=0
    )

def get_income():
    '''
    Returns pandas dataframe containing IRS income data
    '''
    return pd.read_csv(
        home_dir + income_fn,
        index_col=[0,1]
    )

#########################################################################
# main, use extracting original data
if __name__ == '__main__':

    # absolute path to download directory of raw data
    source_dir = path.sep.join(home_dir.split(path.sep)[:3]) + '{0}Downloads{0}'.format(path.sep)

    # original source file names
    atlas_fn = 'DataDownload.xls'
    irs08_fn = '08zpall.csv'
    irs13_fn = 'zipcodeagi13.csv'

    # grab health data from food atlas
    health_df = pd.read_excel(
        source_dir + atlas_fn, 
        sheet_name='HEALTH',
        header=0,
        index_col=0,
        usecols='A,D:G'
    )

    # grab income data from IRS statistics
    income08_df = pd.read_csv(
        source_dir + irs08_fn,
        header=0,
        names=['zipcode', 'agi_bracket', 'total_returns', 'joint_returns', 'exemptions', 'dependents', 'agi_kUSD'],
        index_col=[0,1],
        usecols=[1,2,3,4,6,7,8]
    )
    income13_df = pd.read_csv(
        source_dir + irs13_fn,
        header=0,
        names=['zipcode', 'agi_bracket', 'total_returns', 'joint_returns', 'exemptions', 'dependents', 'agi_kUSD'],
        index_col=[0,1],
        usecols=[2,3,4,6,9,10,11]
    )

    # combine income data
    income_df = income08_df.join(income13_df, how='inner', lsuffix='_08', rsuffix='_13')

    # save extracted data
    health_df.to_csv(home_dir + health_fn)
    income_df.to_csv(home_dir + income_fn)
