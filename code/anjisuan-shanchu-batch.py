#anjisuan-shanchu-batch.py
# Deleting amino acid sequences at random positions
#Author: Sihua Peng, PhD
#Date: 06/07/2023

import csv
import random

n = 10  # Number of batch operations

# Reading CSV file
with open('FJ694790_codon_sequences-new.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Batch operations
for i in range(n):
    # Copy data so that each delete operation is performed on a copy of the original data
    data_modified = [row[:] for row in data]

    # Randomly delete elements
    for _ in range(40):
        row_index = random.randint(0, len(data_modified) - 1)
        row = data_modified[row_index]
        if row:
            index = random.randint(0, len(row) - 1)
            row.pop(index)
            if not row:
                data_modified.pop(row_index)
            else:
                for j in range(row_index + 1, len(data_modified)):
                    data_modified[j-1] = data_modified[j]
                data_modified.pop()

    # Save the results to a separate file
    filename = f'FJ694790_Shanchu-3-hou-{i + 1}.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_modified)



