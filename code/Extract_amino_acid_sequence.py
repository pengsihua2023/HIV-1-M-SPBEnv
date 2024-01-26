
# Extract the amino acid sequence except the start codon and stop codon
 
import csv
from Bio import SeqIO

# Read FASTA files
fasta_file = "L02317.fasta"
records = SeqIO.parse(fasta_file, "fasta")

# Extract codon sequence
codon_sequences = []
for record in records:
    sequence = str(record.seq)
    start_index = 3  # Start codon index
    end_index = len(sequence) - 3  # Stop codon index
    codon_sequence = [sequence[i:i+3] for i in range(start_index, end_index, 3)]
    codon_sequences.extend(codon_sequence)

#Write to CSV file
csv_file = "L02317_codon_sequences.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["serial number", "codon sequence"])
    for i, codon in enumerate(codon_sequences, start=1):
        writer.writerow([i, codon])
    
        
