# Random insertion of amino acid sequences
import csv
import random

def insert_amino_acid(input_file, output_file, amino_acid_file, n1, n):
    # Read the amino acid list
    with open(amino_acid_file, 'r') as f:
        reader = csv.reader(f)
        amino_acids = [row[0] for row in reader]

    # Read the original data file
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        sequences = list(reader)

    # Execute batch insertion n times
    n=2
    for i in range(n):
        # Copy the original sequence
        new_sequences = sequences[:]

        # Perform random insertion n1 times
        n1=40
        for _ in range(n1):
            # Randomly select an amino acid
            amino_acid = random.choice(amino_acids)

            # Randomly select the position to insert
            position = random.randint(0, len(new_sequences))

            # Insert the amino acid at the selected position
            new_sequences.insert(position, [amino_acid])

        # Save the result file
        output_filename = f"{output_file}_{i+1}.csv"
        with open(output_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_sequences)

        print(f"Operation {i+1}/{n} completed, results saved as {output_filename}")

# Input parameters AY173956; FJ694790; FJ817365
input_file = "HQ385448_codon-seq.csv"
#input_file = "KJ158428-NP-11_codon-seq.csv"
output_file = "HQ385448-Charu-hou"
amino_acid_file = "HIV_codon_table-new.csv"
n1 = 40
n = 10

# Call the function to perform the insertion operation
insert_amino_acid(input_file, output_file, amino_acid_file, n1, n)










