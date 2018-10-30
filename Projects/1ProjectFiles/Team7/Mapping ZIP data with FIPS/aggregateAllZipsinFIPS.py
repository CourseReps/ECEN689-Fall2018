import pandas as pd
import numpy as np

def aggregate_zips(input_filename, output_filename):

    taken_df = pd.read_csv(input_filename)

    state_old = taken_df['STATE'].values.tolist()
    agi_stub_old = taken_df['agi_stub'].values.tolist()
    N1_old = taken_df['N1'].values.tolist()
    A1_old = taken_df['A00100'].values.tolist()
    FIPS_old = taken_df['FIPS'].values.tolist()

    FIPS_unique = np.unique(FIPS_old)

    state_new = []
    dependents_no = []
    low_income_no = []
    middle_income_no = []
    high_income_no = []
    dependents_tot = []
    low_income_tot = []
    middle_income_tot = []
    high_income_tot = []
    agi = []
    index = 0
    i=0

    for FIPS in FIPS_unique:

        for n in range(len(FIPS_old)):
            if FIPS == FIPS_old[n]:
                index = n
                break

        state_new.append(state_old[index])
        dependents_no.append(0)
        low_income_no.append(0)
        middle_income_no.append(0)
        high_income_no.append(0)
        dependents_tot.append(0.0)
        low_income_tot.append(0.0)
        middle_income_tot.append(0.0)
        high_income_tot.append(0.0)

        for num in range(len(FIPS_old)):

            if FIPS_old[num] == FIPS:

                if agi_stub_old[num] == 1:
                    dependents_no[i] += N1_old[num]
                    dependents_tot[i] += (N1_old[num]*A1_old[num])

                if agi_stub_old[num] == 2:
                    low_income_no[i] += N1_old[num]
                    low_income_tot[i] += (N1_old[num]*A1_old[num])

                if (agi_stub_old[num] == 3) | (agi_stub_old[num] == 4) | (agi_stub_old[num] == 5):
                    middle_income_no[i] += N1_old[num]
                    middle_income_tot[i] += (N1_old[num]*A1_old[num])

                if (agi_stub_old[num] == 6) | (agi_stub_old[num] == 7) | (agi_stub_old[num] == 8):
                    high_income_no[i] += N1_old[num]
                    high_income_tot[i] += (N1_old[num]*A1_old[num])

        i = i+1
        print(i)

    new_df = pd.DataFrame()
    new_df['FIPS'] = FIPS_unique
    new_df['STATE'] = state_new
    new_df['NO_OF_DEPENDENTS'] = dependents_no
    new_df['NO_OF_LOW_INCOME']  = low_income_no
    new_df['NO_OF_MIDDLE_INCOME'] = middle_income_no
    new_df['NO_OF_HIGH_INCOME'] = high_income_no
    new_df['DEP_TOT'] = dependents_tot
    new_df['LOW_TOT'] = low_income_tot
    new_df['MID_TOT'] = middle_income_tot
    new_df['HIG_TOT'] = high_income_tot
    new_df['AVG_DEPENDENTS'] = round(new_df['DEP_TOT']/new_df['NO_OF_DEPENDENTS'])
    new_df['AVG_LOW_INCOME'] = round(new_df['LOW_TOT']/new_df['NO_OF_LOW_INCOME'])
    new_df['AVG_MIDDLE_INCOME'] = round(new_df['MID_TOT']/new_df['NO_OF_MIDDLE_INCOME'])
    new_df['AVG_HIGH_INCOME'] = round(new_df['HIG_TOT']/new_df['NO_OF_HIGH_INCOME'])
    new_df['AGI'] = round((new_df['DEP_TOT'] + new_df['LOW_TOT'] + new_df['MID_TOT'] + new_df['HIG_TOT']))
    new_df['AGI'] = round(new_df['AGI']/(new_df['NO_OF_DEPENDENTS'] + new_df['NO_OF_LOW_INCOME'] + new_df['NO_OF_MIDDLE_INCOME'] + new_df['NO_OF_HIGH_INCOME']))
    new_df = new_df.drop(['DEP_TOT','LOW_TOT','MID_TOT','HIG_TOT'],axis=1)

    new_df.to_csv(output_filename,index=False)
