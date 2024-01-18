import csv

def find_nonsynonymous_mutations(input_file, output_file):
    codon_table_hiv1 = {
        # [The rest of the codon table is omitted for brevity]
    }
    
    # Open input and output files
    with open(input_file, 'r') as csv_input, open(output_file, 'w', newline='') as csv_output:
        reader = csv.reader(csv_input)
        writer = csv.writer(csv_output)
        
        # Write the header row to the output file
        writer.writerow(['Index', 'Original Codon', 'Nonsynonymous Mutated Codon'])
        
        # Iterate through each row of the CSV file
        for row in reader:
            index = row[0]  # Index
            codon = row[1]  # Codon
            
            # Find nonsynonymous mutated codon in the codon table
            nonsynonymous_mutation = find_nonsynonymous_mutation(codon, codon_table_hiv1)
            
            # Write the result to the output file
            writer.writerow([index, codon, nonsynonymous_mutation if nonsynonymous_mutation else codon])

def find_nonsynonymous_mutation(codon, codon_table):
    # Get the amino acid corresponding to the codon
    amino_acid = codon_table.get(codon, None)
    
    if amino_acid is not None:
        # Find nonsynonymous mutated codons
        nonsynonymous_codons = [k for k, v in codon_table.items() if v != amino_acid]
        
        if nonsynonymous_codons:
            # Return the first nonsynonymous mutated codon found
            return nonsynonymous_codons[0]
    
    return ""

# Define function for batch processing
def batch_find_nonsynonymous_mutations(input_csv, output_dir):
    with open(input_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            input_file_name = row[0]
            output_file_name = input_file_name.replace(".csv", "_Nonsynonymous_Mutation.csv")
            input_file_path = "./" + input_file_name  # Assuming CSV files are in the same directory as this script
            output_file_path = output_dir + output_file_name
            
            # Call the processing function to find nonsynonymous mutations and save the results to the output file
            find_nonsynonymous_mutations(input_file_path, output_file_path)

# Call the function, passing in the CSV file containing input file names and the path of the output directory
input_csv = "Fasta-codon-D-files.csv"
output_dir = "D:/AI-code/"  # Output directory path, ensure this directory exists
batch_find_nonsynonymous_mutations(input_csv, output_dir)

print("Batch operation completed.")

