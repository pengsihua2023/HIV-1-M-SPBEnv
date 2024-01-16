#Author: Sihua Peng, PhD
#Date： 05/29/2023

import csv

def find_synonymous_mutations(input_file, output_file):
    codon_table_hiv1 = {
        # ... (the codon table is unchanged)
    }
    
    # Open input and output files
    with open(input_file, 'r') as csv_input, open(output_file, 'w', newline='') as csv_output:
        reader = csv.reader(csv_input)
        writer = csv.writer(csv_output)
        
        # Write header row to output file
        writer.writerow(['Index', 'Original Codon', 'Synonymous Mutation'])
        
        # Iterate over each row in the CSV file
        for row in reader:
            index = row[0]  # Index
            codon = row[1]  # Original Codon
            
            # Find synonymous mutations in the codon table
            synonymous_mutation = find_synonymous_mutation(codon, codon_table_hiv1)
            
            # Write the result to the output file
            writer.writerow([index, codon, synonymous_mutation if synonymous_mutation else codon])

def find_synonymous_mutation(codon, codon_table):
    # Get the amino acid corresponding to the codon
    amino_acid = codon_table.get(codon, None)
    
    if amino_acid is not None:
        # Find synonymous mutations
        synonymous_codons = [k for k, v in codon_table.items() if v == amino_acid]
        
        # Exclude the original codon
        synonymous_codons = [c for c in synonymous_codons if c != codon]
        
        if synonymous_codons:
            # Return the first synonymous mutation
            return synonymous_codons[0]
    
    return ""

# Function defined for batch operations
def batch_find_synonymous_mutations(input_csv, output_dir):
    with open(input_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            input_file_name = row[0]
            output_file_name = input_file_name.replace(".csv", "_Synonymous_Mutations.csv")
            input_file_path = "./" + input_file_name  # Assuming CSV file is in the same directory as the input files
            output_file_path = output_dir + output_file_name
            
            # Call the processing function to find synonymous mutations and save results to output file
            find_synonymous_mutations(input_file_path, output_file_path)

# Call the function, passing in the CSV file with input file names and the output directory path
input_csv = "Fasta-codon-files.csv"
output_dir = "./output/"  # Ensure this output directory exists
batch_find_synonymous_mutations(input_csv, output_dir)

print("Bulk operation completed.")

