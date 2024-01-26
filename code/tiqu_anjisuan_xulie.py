
 
 
import csv
from Bio import SeqIO

# 读取FASTA文件
fasta_file = "L02317.fasta"
records = SeqIO.parse(fasta_file, "fasta")

# 提取密码子序列
codon_sequences = []
for record in records:
    sequence = str(record.seq)
    start_index = 3  # 起始密码子索引
    end_index = len(sequence) - 3  # 终止密码子索引
    codon_sequence = [sequence[i:i+3] for i in range(start_index, end_index, 3)]
    codon_sequences.extend(codon_sequence)

# 写入CSV文件
csv_file = "L02317_codon_sequences.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["序号", "密码子序列"])
    for i, codon in enumerate(codon_sequences, start=1):
        writer.writerow([i, codon])
    
        
