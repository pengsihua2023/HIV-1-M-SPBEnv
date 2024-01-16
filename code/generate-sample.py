#Author: Sihua Peng, PhD
#Date： 07/25/2023

import random
from Bio import SeqIO, Seq
import pandas as pd

# Read the CSV file containing filenames
csv_file = "L-segment.csv"
file_data = pd.read_csv(csv_file, header=None)
fasta_files = file_data[0].tolist()

# Initialize a list to store selected sequences
selected_sequences = []

# Iterate through each FASTA file
for fasta_file in fasta_files:
    records = list(SeqIO.parse(fasta_file, "fasta"))

    if len(records) >= 4:
        # Randomly select one sequence from the current FASTA file
        selected_record = random.choice(records)
        selected_sequences.append(selected_record.seq)
    else:
        print(f"Skipping {fasta_file} due to insufficient sequences")

# Merge the selected sequences into one DNA sequence
merged_sequence = Seq.Seq("".join(str(seq) for seq in selected_sequences))

# Save the merged sequence to a new FASTA file
output_file = "Generated-sample-10.fas"
with open(output_file, "w") as output_handle:
    record = SeqIO.SeqRecord(merged_sequence, id="MergedSequence", description="Merged DNA sequence")
    SeqIO.write(record, output_handle, "fasta")

print(f"Merged DNA sequence saved to {output_file}")


