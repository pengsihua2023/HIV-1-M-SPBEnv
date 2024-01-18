import csv

def find_synonymous_mutations(input_file, output_file):
    codon_table_hiv1 = {
           'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AGA': 'R', 'AGG': 'R', 'CGC': 'R', 'CGG': 'R',
        'AAT': 'N', 'AAC': 'N', 'GAT': 'D', 'GAC': 'D',
        'TGT': 'C', 'TGC': 'C', 'CAA': 'Q', 'CAG': 'Q',
        'GAA': 'E', 'GAG': 'E', 'GGT': 'G', 'GGC': 'G',
        'GGA': 'G', 'GGG': 'G', 'CAT': 'H', 'CAC': 'H',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L',
        'CTA': 'L', 'CTG': 'L', 'AAA': 'K', 'AAG': 'K',
        'TTT': 'F', 'TTC': 'F', 'CCT': 'P', 'CCC': 'P',
        'CCA': 'P', 'CCG': 'P', 'TCT': 'S', 'TCC': 'S',
        'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'TGG': 'W', 'TAT': 'Y', 'TAC': 'Y', 'GTT': 'V',
        'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'TAA': '*',
        'TAG': '*', 'TGA': '*'
    }
    
    # Open the input and output files
    with open(input_file, 'r') as csv_input, open(output_file, 'w', newline='') as csv_output:
        reader = csv.reader(csv_input)
        writer = csv.writer(csv_output)
        
        # Write the header row to the output file
        writer.writerow(['Index', 'Original Codon', 'Synonymous Mutation'])
        
        # Iterate through each row of the CSV file
        for row in reader:
            index = row[0]  # Index
            codon = row[1]  # Codon
            
            # Find synonymous mutations in the codon table
            synonymous_mutation = find_synonymous_mutation(codon, codon_table_hiv1)
            
            # Write the result to the output file
            writer.writerow([index, codon, synonymous_mutation if synonymous_mutation else codon])

def find_synonymous_mutation(codon, codon_table):
    # Get the amino acid corresponding to the codon
    amino_acid = codon_table.get(codon, None)
    
    if amino_acid is not None:
        # Find synonymous codons
        synonymous_codons = [k for k, v in codon_table.items() if v == amino_acid]
        
        # Exclude the original codon
        synonymous_codons = [c for c in synonymous_codons if c != codon]
        
        if synonymous_codons:
            # Return the first synonymous codon found
            return synonymous_codons[0]
    
    return ""

# Define a function for batch processing
def batch_find_synonymous_mutations(input_csv, output_dir):
    with open(input_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            input_file_name = row[0]
            output_file_name = input_file_name.replace(".csv", "_Synonymous_Mutation.csv")
            input_file_path = "./" + input_file_name  # Assume CSV files are in the same directory as the script
            output_file_path = output_dir + output_file_name
            
            # Call the processing function to find synonymous mutations and save the results to the output file
            find_synonymous_mutations(input_file_path, output_file_path)

# Call the function with the CSV file containing input file names and the path of the output directory
input_csv = "Fasta-codon-D-files.csv"
output_dir = "D:/AI-code/"  # Output directory path, ensure this directory exists
batch_find_synonymous_mutations(input_csv, output_dir)

print("Bulk operation completed.")
