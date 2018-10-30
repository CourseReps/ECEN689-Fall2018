from addFIPStoIncome import map_fips
from FinalIncomeForCounty import AggregateDatainFIPS
from aggregateAllZipsinFIPS import aggregate_zips
from MapIncomeAndHealth import map_function

map_fips('08zpall.csv', 'ZIP-COUNTY-FIPS_2017-06.csv', '2008_income_with_FIPS.csv')
AggregateDatainFIPS('2008_income_with_FIPS.csv', '2008_FIPS_income_final.csv')
aggregate_zips('2008_FIPS_income_final.csv', '2008_income_seperated.csv')
map_function('2008_income_seperated.csv', 'Diabeties_obese.csv', '2008_mapped_data.csv')

map_fips('13zpallagi.csv', 'ZIP-COUNTY-FIPS_2017-06.csv', '2013_income_with_FIPS.csv')
AggregateDatainFIPS('2013_income_with_FIPS.csv', '2013_FIPS_income_final.csv')
aggregate_zips('2013_FIPS_income_final.csv', '2013_income_seperated.csv')
map_function('2013_income_seperated.csv', 'Diabeties_obese.csv', '2013_mapped_data.csv')
