#==========================================#
# This code was written by Sihua Peng, PhD.
#==========================================#

import random
from Bio import SeqIO, Seq
import pandas as pd

# Read the CSV file containing filenames
csv_file = "L-segment-10-seg.csv"
file_data = pd.read_csv(csv_file, header=None)
fasta_files = file_data[0].tolist()

num_samples = 100  # Number of random samples you want to generate

# List to store all generated SeqRecords
all_records = []

for sample_num in range(1, num_samples + 1):

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

    # Create a new SeqRecord for the merged sequence and add it to the list
    #record = SeqIO.SeqRecord(merged_sequence, id=f"MergedSequence_{sample_num}", description=f"Generate-L #{sample_num}")
    record = SeqIO.SeqRecord(merged_sequence, id=f"Generated_L_{sample_num}", description=f"")
    #record = SeqIO.SeqRecord(merged_sequence, id=f"MergedSequence_{sample_num}")
    all_records.append(record)

    print(f"Merged DNA sequence for sample {sample_num} created.")

# Save all the records to a single FASTA file
output_file = "Generated-L-samples-0925.fas"
with open(output_file, "w") as output_handle:
    SeqIO.write(all_records, output_handle, "fasta")

print(f"All merged DNA sequences saved to {output_file}")
