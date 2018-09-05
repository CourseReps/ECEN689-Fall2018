import pandas as pd

# Read file.
student_identity = 'blthayer'
data_dir = '../../Challenges/1Files'
filename = data_dir + '/1challenge1activity_' + student_identity + '.csv'
df = pd.read_csv(filename)

# Define columns we'll be using.
cols = ['Sample ' + str(x) for x in range(12)]

# Compute mean and variance across columns
m = df[cols].mean(axis=1)
v = df[cols].var(axis=1)

# Assign.
df['Mean'] = m
df['Variance'] = v

# Remove the 'Unnamed' column
df = df.drop('Unnamed: 0', axis=1)

# Write.
df.to_csv(filename)
