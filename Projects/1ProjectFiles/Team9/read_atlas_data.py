########################################################################
# IMPORTS

# Installed packages:
import pandas as pd

# Standard Library:
import os.path

########################################################################
# CONSTANTS

# File paths:
DATA_DIR = 'datasets'
HEALTH_DATA_COUNTY_FILE = \
    os.path.join(DATA_DIR, 'Food_Atlas_County_2013.csv')
HEALTH_DATA_STATE_FILE = \
    os.path.join(DATA_DIR, 'Food_Atlas_State_2013.csv')

# Data types for the county data.
COUNTY_DTYPES = {'FIPS': str, 'State': str, 'County': str,
                 'Population Estimate, 2013': float, 'VLFOODSEC_13_15': float,
                 'FOODINSEC_13_15': float, 'PCT_DIABETES_ADULTS13': float,
                 'PCT_OBESE_ADULTS13': float}

########################################################################
# FUNCTIONS


def read_data():
    """Function to simply read the atlas data from file."""
    health_county_data = pd.read_csv(HEALTH_DATA_COUNTY_FILE,
                                     dtype=COUNTY_DTYPES)

    health_state_data = pd.read_csv(HEALTH_DATA_STATE_FILE)
    # print(health_county_data.head(),health_state_data.head())

    # Return.
    return health_county_data, health_state_data


########################################################################
# MAIN


if __name__ == '__main__':
    read_data()