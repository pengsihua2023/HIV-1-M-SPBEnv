
# This code was written by Sihua Peng, PhD.

from Bio import SeqIO
import csv

# Define a function to divide the DNA sequence into N segments
def split_sequence(dna_sequence):
    segments = [
        dna_sequence[0:300],
        dna_sequence[300:600],
        dna_sequence[600:900],
        dna_sequence[900:1200],
        dna_sequence[1200:1500],        
        dna_sequence[1500:1800],
        dna_sequence[1800:2100],
        dna_sequence[2100:2400],
        dna_sequence[2400:]
 #      dna_sequence[2700:]
     #  dna_sequence[3000:]
    ]
    
    return segments

# Clear CSV file
def clear_csv_files():
    output_files = [
        'L-first-segment.csv',
        'L-second-segment.csv',
        'L-third-segment.csv',
        'L-fourth-segment.csv',
        'L-fifth-segment.csv',        
        'L-sixth-segment.csv',
        'L-seventh-segment.csv',
        'L-eighth-segment.csv',
        'L-nineth-segment.csv'
#       'A2-tenth-segment.csv'
     #  'H-eleventh-segment.csv'
    ]
    for file in output_files:
        with open(file, 'w') as csvfile:
            pass

#Clear CSV file
clear_csv_files()

# Read the L-3plus1.csv file and process each fasta format file
with open('L-3plus1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        fasta_file = row[0] if row else ""  # Use the value of the row as the filename
        if fasta_file:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                dna_sequence = str(record.seq)
                segments = split_sequence(dna_sequence)

                # Write each DNA sequence into the corresponding CSV file
                output_files = [
                    'L-first-segment.csv',
                    'L-second-segment.csv',
                    'L-third-segment.csv',
                    'L-fourth-segment.csv',
                    'L-fifth-segment.csv',
                    'L-sixth-segment.csv',
                    'L-seventh-segment.csv',
                    'L-eighth-segment.csv',
                    'L-nineth-segment.csv'
 #                  'A2-tenth-segment.csv'
                #   'H-eleventh-segment.csv'
                    
                ]
                for j, segment in enumerate(segments):
                    with open(output_files[j], 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([segment])  # Write fragments to rows as a single-element list






