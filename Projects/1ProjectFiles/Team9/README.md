# Project 1, Team 9
by 
* Brandon
* Jatin
* Harish
* Swati


## Setup
1. Download the zip file in the "ZIP Code Data" section on the [IRS website](https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2013-zip-code-data-soi)
2. Copy the zip archive `zipcode2013.zip` to this directory.
3. Extract into `zipcode2013` in this directory.

### requirements
* numpy
* pandas
* matplotlib
* plotly
* sklearn
* scipy
* seaborn

####Plotly Choropleth maps require the following:
* geopandas==0.3.0
* pyshp==1.2.10
* shapely==1.6.3
# NOTE: Had to run sudo apt-get install libgeos-dev
# NOTE: Had to downgrade from Python3.7 to Python3.6


### .gitignore
.gitignore file to avoid filling up the repository with junk.

## Final Report FileName
 * 

## Data Sources
 * CDC
 * IRS
 * ATLAS

## Python Code
 * fipsZipHandler.py  ---- Module to convert fips to zips and Vice Vers
 * read_irs.py        ---- Reads the IRS 2013 data
 * read_atlas_data.py ---- Reads atlas data 
 * data_extraction_county.py ---- extracts the health data county/state wise
 * analysis.py        ---- Responsible for final visualisation plots(scatter/US map plot)

 ## Jupyter Folder Files
 * Correaltion_heatmap_generate ---- generates correlation heatmap on the final data
 * visualisation_cdc            ---- analyses data from cdc


