import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 读取FASTA文件中的序列，并返回序列及其ID
def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq).upper())
    return sequences

# 计算序列的k-mer频率
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

# 直接读取CSV文件中的标签
def read_labels(file_path):
    labels_df = pd.read_csv(file_path, header=None)  # 假设CSV文件没有列名
    labels = labels_df[0].tolist()  # 假设标签在第一列
    return labels

# 创建一个颜色映射字典，为每种标签分配不同的颜色
def get_color_map(labels):
    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}
    return label_to_color

# 主要流程
sequences = read_fasta('sihua.fasta')
labels = read_labels('Label.csv')  # 直接按顺序分配标签
kmer_freqs = get_kmer_freqs(sequences, k=6)

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
