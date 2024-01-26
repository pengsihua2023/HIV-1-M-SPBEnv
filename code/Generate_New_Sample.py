# Merge rows of nucleotides and add a start codon at the beginning and a stop codon at the end
import csv

def merge_strings(csv_file, output_file):
    # Read CSV file
    with open(csv_file, 'r', encoding='utf-8-sig') as file:  # Specify encoding as utf-8-sig
        reader = csv.reader(file)
        data = [row[0].strip() for row in reader]  # Remove whitespace characters

    # Merge strings
    merged_string = ''.join(data)

    # Add "ATG" at the beginning and "TAA" at the end
    merged_string = "ATG" + merged_string + "TAA"

    # Save as FASTA format file
    with open(output_file, 'w', encoding='utf-8') as file:  # Specify encoding as utf-8
        file.write('>merged_string\n')
        file.write(merged_string)

# Specify the paths for the input CSV file and the output FASTA file
csv_file = 'KC595165_tubianhou-10.csv'
output_file = 'KC595165-10-hebing.fasta'

# Call function to merge strings and save
merge_strings(csv_file, output_file)
