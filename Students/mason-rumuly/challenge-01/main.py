import pandas as pd
import numpy as np

# student ID
student_identity = 'masondatminer'

# load operative file
filename = 'Challenges/1Files/1challenge1activity_{}.csv'.format(student_identity)
sample_df = pd.read_csv(filename)

# metadata
samples = list(sample_df)[3:]

# iterate over rows
for row in range(sample_df.shape[0]):
    # estimate mean with sample mean
    sample_df.loc[row, 'Mean'] = sum([
        sample_df[sn][row] for sn in samples
    ]) / len(samples)

    # estimate variance (unbiased estimator)
    sample_df.loc[row, 'Variance'] = sum([
        pow(sample_df[sn][row] - sample_df.loc[row, 'Mean'], 2) for sn in samples
    ]) / (len(samples) - 1)

# save results
sample_df.to_csv(filename, sep=',', encoding='utf-8')