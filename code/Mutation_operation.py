# After obtaining synonymous mutations, non-synonymous mutations, insertion mutations, and deletion mutations in the previous step, perform a two-column data swap to implement the mutation operation.
# Function: Swap two columns of data to implement mutation operation
# Author: Sihua Peng, PhD
import csv

#AY173956,FJ694790,FJ817365
# Read z_values.csv file
with open('AY173956_3_ftytb-suijishu.csv', 'r') as z_file:
    reader = csv.reader(z_file)
    z_data = list(reader)

# Read the mutation csv file
with open('AY173956_ftytb.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Iterate through rows of z_values data
for i, z_row in enumerate(z_data):
    z_values = [int(z) for z in z_row]

    # Copy data for mutation operation
    mutated_data = [row[:] for row in data]

    # Iterate through the list of z values, replacing characters in the corresponding cells of the first column in the FJ817365_tytb.csv file
    for z in z_values:
        row_index = z - 1
        if row_index < len(mutated_data):
            mutated_data[row_index][0] = mutated_data[row_index][1]

    # Save the results after mutation to a separate file
    filename = f'AY173956-ftytb-3-hou-{i + 1}.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in mutated_data:
            writer.writerow([row[0]])

    # Restore original data for the next mutation operation
    for z in z_values:
        row_index = z - 1
        if row_index < len(data):
            data[row_index][0] = data[row_index][1]
