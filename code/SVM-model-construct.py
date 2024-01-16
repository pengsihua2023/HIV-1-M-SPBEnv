#Author: Sihua Peng, PhD
#Date: 06/09/2023

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import svm
import joblib

# Read fasta format file
sequences = []
# Various fasta files to parse (uncomment as necessary)
#for record in SeqIO.parse('HIV_ABCDFG_534_shengcheng_data.fasta', 'fasta'):
#for record in SeqIO.parse('HIV_ABCDFG_534_ftytb.fasta', 'fasta'):
#for record in SeqIO.parse('HIV_ABCDFG_534_Charu.fasta', 'fasta'):
#for record in SeqIO.parse('HIV_ABCDFG_534_Shanchu_TB.fasta', 'fasta'):
#for record in SeqIO.parse('HIV_ABCDFG_534_Four_in_One.fasta', 'fasta'):
#for record in SeqIO.parse('HIV_ABCDFG_541_three_samples.fasta', 'fasta'):
#for record in SeqIO.parse('HIV_ABCDFG_534_one_sample-2.fasta', 'fasta'):
#for record in SeqIO.parse('HIV_add-L-534.fasta', 'fasta'):
#for record in SeqIO.parse('HIV_ABCDFGL_623.fasta', 'fasta'):
for record in SeqIO.parse('HIV_12-class-new.fasta', 'fasta'):
    sequences.append(str(record.seq))

# Define k-mer size and feature dimension
k = 6
feature_dim = 4 ** k

# Create feature vector
def create_feature_vector(sequence, k):
    feature_vector = np.zeros(4 ** k)
    for i in range(len(sequence)-k+1):
        kmer = sequence[i:i+k]
        index = kmer_to_index(kmer)
        feature_vector[index] += 1
    return feature_vector

# Convert k-mer to index for feature vector
def kmer_to_index(kmer):
    index = 0
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, base in enumerate(kmer[::-1]):
        index += mapping[base] * (4 ** i)
    return index

# Create feature matrix
feature_matrix = np.zeros((len(sequences), feature_dim))
for i, sequence in enumerate(sequences):
    feature_matrix[i] = create_feature_vector(sequence, k)

# Read label file (assuming label file is in CSV format)
labels = pd.read_csv('Labels_1068-12-class.csv')

# Extract feature and label data
X = feature_matrix
y = labels['Label'].values

# Create SVM classifier model
svm_model = SVC(gamma='scale')

# Train the model
svm_model.fit(X, y)

# Save the trained model
joblib.dump(svm_model, 'svm_model-all-12-class-2.pkl')
