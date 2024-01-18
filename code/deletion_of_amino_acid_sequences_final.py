# Randomly delete some amino acid sequences, batch operation
# Author: Sihua Peng, PhD
import csv
import random

n = 2  # Number of batch operations

# Read CSV file: KJ158428-NP-11_codon-seq
with open('HQ385448_codon-seq.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Batch operations
for i in range(n):
    # Copy data to perform each deletion operation on a copy of the original data
    data_modified = [row[:] for row in data]

    # Randomly delete elements. Number of mutated amino acids m
    m=80
    for _ in range(m):
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

    # Save the results to a separate file, the ending number indicates the operation count
    filename = f'HQ385448_Shanchu-hou-{i + 1}.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_modified)

