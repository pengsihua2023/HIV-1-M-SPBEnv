#Author: Sihua Peng, PhD
#Date: 06/28/2023
from Bio import SeqIO
import csv

# Define function to split DNA sequence into n segments
def split_sequence(dna_sequence):
    segments = [
        dna_sequence[0:250],
        dna_sequence[250:500],
        dna_sequence[500:750],
        dna_sequence[750:1000],
        dna_sequence[1000:1250],        
        dna_sequence[1250:1500],
        dna_sequence[1500:1750],
        dna_sequence[1750:2000],
        dna_sequence[2000:2250],
        dna_sequence[2250:]
    ]
    
    return segments

# Clear CSV files
def clear_csv_files():
    output_files = [
        'K-first-segment.csv',
        'K-second-segment.csv',
        'K-third-segment.csv',
        'K-fourth-segment.csv',
        'K-fifth-segment.csv',        
        'K-sixth-segment.csv',
        'K-seventh-segment.csv',
        'K-eighth-segment.csv',
        'K-nineth-segment.csv',
        'K-tenth-segment.csv'
    ]
    for file in output_files:
        with open(file, 'w') as csvfile:
            pass

# Clear CSV files
clear_csv_files()

# Read sihua.csv file and process each fasta format file
with open('DNA_file-5.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        fasta_file = row[0] if row else ""  # Use the value of the row as the filename
        if fasta_file:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                dna_sequence = str(record.seq)
                segments = split_sequence(dna_sequence)

                # Write each segment of DNA sequence to the corresponding CSV file
                output_files = [
                    'K-first-segment.csv',
                    'K-second-segment.csv',
                    'K-third-segment.csv',
                    'K-fourth-segment.csv',
                    'K-fifth-segment.csv',
                    'K-sixth-segment.csv',
                    'K-seventh-segment.csv',
                    'K-eighth-segment.csv',
                    'K-nineth-segment.csv',
                    'K-tenth-segment.csv'
                ]
                for j, segment in enumerate(segments):
                    with open(output_files[j], 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([segment])  # Write the segment as a single-element list into the row






