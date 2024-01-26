# Determine whether the sequence is an open reading frame, that is, the first three bases are the start codons, 
# the last three bases are the stop codons, and whether the middle DNA sequence encodes amino acids according to the triplet code.
from Bio import SeqIO

def is_start_codon(seq):
    start_codons = ["ATG", "GTG", "TTG"]
    return seq[:3] in start_codons

def is_stop_codon(seq):
    stop_codons = ["TAA", "TAG", "TGA"]
    return seq[-3:] in stop_codons

def is_multiple_of_three(seq):
    return len(seq) % 3 == 0

def translate_codon(codon):
    codon_table = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"
    }
    return codon_table.get(codon, "X")

def translate_sequence(seq):
    protein_seq = ""
    for i in range(0, len(seq)-2, 3):
        codon = seq[i:i+3]
        protein_seq += translate_codon(codon)
    return protein_seq

def is_valid_amino_acid_sequence(seq):
    protein_seq = translate_sequence(seq)
    return "*" not in protein_seq

def is_complete_orf(seq):
    if is_start_codon(seq) and is_stop_codon(seq[-3:]) and is_multiple_of_three(seq) and is_valid_amino_acid_sequence(seq[3:-3]):
        return True
    return False

def process_fasta_file(filename):
    with open(filename, 'r') as file:
        sequences = SeqIO.parse(file, 'fasta')

        with open("L02317-yes.fasta", 'w') as output_file:
            for seq_record in sequences:
                seq = str(seq_record.seq)
                if is_complete_orf(seq):
                    output_file.write(f">{seq_record.id}\n")
                    output_file.write(f"{seq}\n")
                    output_file.write("------\n")




# operation
process_fasta_file("L02317.fasta")





