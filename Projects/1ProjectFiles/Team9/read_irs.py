"""
Module for reading IRS data. To be consistent with health data, we're
using 2013 data.

As described in the README.md file, this module expects you to have a
folder called 'zipcode2013' in this directory. Within this directory,
at the minimum a file called '13zpallagi.csv'

NOTE: The IRS data excludes those with a gross deficit
(rather than income).
"""
########################################################################
# IMPORTS

# Installed packages:
import numpy as np
import pandas as pd

# Standard Library:
import os.path

# Project modules:
from fipsZipHandler import FipsZipHandler

########################################################################
# CONSTANTS

# File paths:
# Folder w/ IRS data in it.
IRS_DIR = 'zipcode2013'
# IRS data file name.
IRS_FILE = 'zipcodeagi13.csv'
# Full path to IRS data file.
IRS_FILE_PATH = os.path.join('.', IRS_DIR, IRS_FILE)

# Columns to read. Keys are columns, values are brief explanations.
COLUMNS = {'STATEFIPS': 'State FIPS code',
           'STATE': 'Two letter state code', 'zipcode': 'Zip code',
           'agi_stub': ('Code for income bracket. 1: $1-$25k, 2: $25k-$50k, '
                        + '3: $50k-$75k, 4: $75k-$100k, 5: $100k-$200k, '
                        + '6: $200k+'),
           'N1': '# of returns',
           'MARS1': '# of single returns',
           'MARS2': '# of joint returns',
           'MARS4': '# of head of household returns',
           'NUMDEP': '# of dependents',
           'A00100': 'Adjusted gross income (AGI)',
           'A02650': 'Total income amount'}

# Describe each agi_stub. Dollar signs are escaped for Tex.
AGI_STUBS = {1: '\$1-\$25k', 2: '\$25k-\$50k', 3: '\$50k-\$75k',
             4: '\$75k-\$100k', 5: '\$100k-\$200k', 6: '\$200k+'}

# Define data types for the IRS data. Codes should be mapped as a string
# to avoid dropping leading 0's.
COLUMN_DTYPES = {'STATEFIPS': np.str,
                 'STATE': np.str,
                 'zipcode': np.str,
                 'agi_stub': np.int64,
                 'N1': np.float64,
                 'MARS1': np.float64,
                 'MARS2': np.float64,
                 'MARS4': np.float64,
                 'NUMDEP': np.float64,
                 'A00100': np.float64,
                 'A02650': np.float64}

########################################################################
# FUNCTIONS


def read_data():
    """Function to simply read the IRS data from file."""
    irs_data = pd.read_csv(IRS_FILE_PATH, header=0,
                           usecols=list(COLUMNS.keys()), dtype=COLUMN_DTYPES)

    # Convert zipcode to a string. Note: it'd be more efficient to
    # define the data type when the file is read, but that can be a real
    # hassle. However, not doing so causes leading zeros to be dropped
    #in the FIPS and zipcodes. The information loss is during data read,
    #So converting the datatype subsequently has no utility.
    #irs_data['zipcode'] = irs_data['zipcode'].astype(str)

    # Convert STATEFIPS to a string. Same efficiency note as above.
    #irs_data['STATEFIPS'] = irs_data['STATEFIPS'].astype(str)

    return irs_data


def lookup_fips(irs_data):
    """Function to associate FIPS codes based on IRS zipcodes"""
    # Initialize FipsZipHandler object
    fz_obj = FipsZipHandler()

    # Translate IRS data zip codes to FIPS codes.
    irs_fips = [fz_obj.getFipsForZipcode(z) for z in irs_data['zipcode']]

    # Add column to irs_data for FIPS code.
    irs_data['FIPS'] = irs_fips

    # Investigate the NaN data.
    nan_data = irs_data[irs_data.isnull().any(axis=1)]

    # Return.
    return irs_data


def aggregate_by_fips(irs_data):
    """Function to combine IRS data by FIPS code.

    NOTE: This doesn't necessarily need to be in a function since pandas
    makes this so easy.
    """
    # Drop NaN state values.
    irs_data.dropna(inplace=True)

    # Use groupby to aggregate.
    aggregated_data = irs_data.groupby(['FIPS', 'agi_stub']).sum()

    # For simplicity, change the multi-index into columns.
    # TODO: We may want to keep the multi-index around?
    aggregated_data.reset_index(inplace=True)

    return aggregated_data


def wealth_per_person(irs_data):
    """Estimate wealth per person with the IRS data.

    Note
    """
    # Single returns + 2 * joint returns + number of dependents.
    # NOTE: It seems that head of household (MARS4) is not mutually
    # exclusive with MARS1. So we'll exclude it.
    irs_data['total_people'] = (irs_data['MARS1'] + 2 * irs_data['MARS2']
                                + irs_data['NUMDEP'])

    # Divide AGI by the number of people.
    # NOTE: It would seem that the IRS AGI number needs to be multiplied
    # by 1000. This is evidenced by taking irs_data['A00100']
    # / irs_data['N1'] and noticing that all the values correctly fall
    # in the irs_data['agi_stub'] categories.
    irs_data['agi_per_person'] = np.rint(1000 * irs_data['A00100']
                                         / irs_data['total_people'])

    # Ensure NaN's are 0'ed out.
    irs_data['agi_per_person'] = irs_data['agi_per_person'].fillna(0)

    return irs_data


def compute_percentages(irs_data):
    """Compute pct of returns and pct of people for each FIPS code."""
    # Sum returns and people by FIPS
    totals = irs_data[['FIPS', 'N1', 'total_people']].groupby(['FIPS']).sum()

    # Join the totals into the irs data
    irs_data = irs_data.join(totals, on='FIPS', rsuffix='_total_for_FIPS')

    # Compute percentages.
    irs_data['N1_pct_of_FIPS'] = irs_data['N1'] / irs_data['N1_total_for_FIPS']
    irs_data['total_people_pct_of_FIPS'] = \
        (irs_data['total_people'] / irs_data['total_people_total_for_FIPS'])

    return irs_data


def get_irs_data():
    """Main function to load, map, and aggregate IRS data.
    """
    # Read file.
    data = read_data()

    # Get FIPS for all zip codes.
    data = lookup_fips(data)

    # Aggregate by FIPS codes.
    data = aggregate_by_fips(data)

    # Compute wealth per person.
    data = wealth_per_person(data)

    # Compute percentages of number of returns and total people.
    data = compute_percentages(data)

    return data

########################################################################
# MAIN


if __name__ == '__main__':
    get_irs_data()
