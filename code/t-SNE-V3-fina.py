import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Read sequences from a FASTA file and return the sequence
def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq).upper())
    return sequences

# Calculate the k-mer frequency of a sequence
def get_kmer_freqs(sequences, k):
    kmer_set = set()
    kmer_freqs = []
    
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_set.add(kmer)
    
    for seq in sequences:
        kmer_freq = {kmer: 0 for kmer in kmer_set}
        total_kmers = len(seq) - k + 1
        for i in range(total_kmers):
            kmer = seq[i:i+k]
            kmer_freq[kmer] += 1
        
        for kmer in kmer_freq:
            kmer_freq[kmer] /= total_kmers
        
        kmer_freqs.append([kmer_freq[kmer] for kmer in kmer_set])
    
    return np.array(kmer_freqs)

# Read tags directly from CSV files
def read_labels(file_path):
    labels_df = pd.read_csv(file_path, header=None)  # Assuming the CSV file has no column names
    labels = labels_df[0].tolist()  # Assume the label is in the first column
    return labels

# Create a color map dictionary to assign a different color to each type of label
def get_color_map(labels):
    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}
    return label_to_color

# Main process
sequences = read_fasta('sihua.fasta')
labels = read_labels('Label.csv')  # Assign tags directly in sequence
kmer_freqs = get_kmer_freqs(sequences, k=7)

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(kmer_freqs)

color_map = get_color_map(labels)

plt.figure(figsize=(8, 5))
for label, color in color_map.items():
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, color=color)
plt.title('t-SNE Visualization of Genome Sequences with Labels')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.show()
