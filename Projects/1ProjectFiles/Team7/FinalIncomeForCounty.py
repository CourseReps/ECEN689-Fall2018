import pandas as pd
import numpy as np


def AggregateDatainFIPS(input_filename, output_filename):

    taken_df = pd.read_csv('2008_income_with_FIPS.csv')

    state_old = taken_df['state'].values.tolist()
    agi_stub_old = taken_df['agi_class'].values.tolist()
    N1_old = taken_df['n1'].values.tolist()
    A1_old = taken_df['a00100'].values.tolist()
    FIPS_old = taken_df['FIPS'].values.tolist()

    FIPS_unique = np.unique(FIPS_old)

    i = 0
    N1_new = []
    A1_new = []
    state_new = []
    FIPS_new = []
    agi_stub_new = []
    index = 0

    print(len(FIPS_unique))
    #try:
    for FIPS in FIPS_unique:

        for n in range(len(FIPS_old)):
            if FIPS == FIPS_old[n]:
                index = n
                break

        for j in range(1,9):
            N1_new.append(0)
            A1_new.append(0)
            agi_stub_new.append(j)
            FIPS_new.append(FIPS)
            state_new.append(state_old[index])


        for num in range(len(FIPS_old)):

            if FIPS_old[num] == FIPS:

                for k in range(1,9):

                    if agi_stub_old[num] == k:
                        N1_new[(i*8)+(k-1)] += N1_old[num]
                        A1_new[(i*8)+(k-1)] += A1_old[num]

        i = i+1
        print(i)

    print(8*len(FIPS_unique))
    print(len(N1_new))
    print(len(A1_new))

    new_df = pd.DataFrame()
    new_df['FIPS'] = FIPS_new
    new_df['STATE'] = state_new
    new_df['agi_stub'] = agi_stub_new
    new_df['N1'] = N1_new
    new_df['A00100'] = A1_new

    new_df.to_csv('2008_FIPS_income_final.csv',index=False)
