import pandas as pd
import numpy as np
from make_final import drop_non_countries

# file names
adjacency_mat_file = 'Students/mason-rumuly/challenge-03/3parameters_countries_only.csv'
node_file = 'Students/mason-rumuly/challenge-03/nodes.csv'
edge_file = 'Students/mason-rumuly/challenge-03/edges.csv'

# import coefficient table
adjacency_mat = pd.read_csv(adjacency_mat_file, index_col='Country Name', encoding='cp1252').dropna(axis=0)
adjacency_mat = drop_non_countries(adjacency_mat)
countries = list(enumerate(adjacency_mat.index.values))

# Enumerate nodes
with open(node_file, 'w') as f:
    f.write('id;label\n')
    for i, c in countries:
        f.write('{};"{}"\n'.format(i, c))

# Enumerate edges
with open(edge_file, 'w') as f:
    f.write('Source;Target;Label\n')
    for source, _ in countries:
        for sink, _ in countries:
            if adjacency_mat.iloc[source, sink] != 0:
                f.write('{};{};"{}"\n'.format(source, sink, adjacency_mat.iloc[source, sink]))
