import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from Bio import SeqIO

# Read fasta format file
sequences = []
for record in SeqIO.parse('HIV_12-class-new.fasta', 'fasta'):
    sequences.append(str(record.seq))

# Define k-mer size and feature dimension
k = 8
feature_dim = 4 ** k

# Create feature vector
def create_feature_vector(sequence, k):
    feature_vector = np.zeros(4 ** k)
    for i in range(len(sequence)-k+1):
        kmer = sequence[i:i+k]
        index = kmer_to_index(kmer)
        feature_vector[index] += 1
    return feature_vector

# Convert k-mer to index in feature vector
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

# Read label file
labels = pd.read_csv('Labels_1068-12-class.csv')  # Assuming label file is in CSV format

# Define label mapping
label_mapping = {'A1': 0, 'A2': 1, 'B': 2, 'C': 3, 'D': 4, 'F1': 5, 'F2': 6, 'G': 7, 'H': 8, 'J': 9, 'K': 10, 'L': 11}

# Map labels to numbers
labels['Label'] = labels['Label'].map(label_mapping)

# Extract feature and label data
X = feature_matrix
y = labels['Label'].values

# Define Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, ae_hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ae_hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(ae_hidden_dim, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return encoded_x

# Define neural network model
# Self-Attention Mechanism
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

# Neural network structure
class DeepModel(nn.Module):                                    
    def __init__(self, input_dim, num_classes, ae_hidden_dim):
        super(DeepModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.attention = SelfAttention(embed_size=ae_hidden_dim // 8, heads=8)  # Adjust embed_size as needed
        self.fc1 = nn.Linear(ae_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.autoencoder = Autoencoder(64 * ((input_dim - 2) // 2), ae_hidden_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(torch.relu(x))
        x = x.view(x.size(0), -1)
        encoded_x = self.autoencoder(x)
        
        # Reshape encoded_x to have a 'sequence length' dimension
        reshaped_x = encoded_x.view(encoded_x.shape[0], 8, -1)
        
        attention_out = self.attention(reshaped_x, reshaped_x, reshaped_x)
        
        # Flatten it back to feed it into the next layer
        attention_out = attention_out.view(attention_out.shape[0], -1)
        
        x = torch.relu(self.fc1(attention_out))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Data preparation
batch_size = 32
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train.astype(np.float32)).unsqueeze(1)
X_test = torch.tensor(X_test.astype(np.float32)).unsqueeze(1)
y_train = torch.tensor(y_train.astype(np.int64))
y_test = torch.tensor(y_test.astype(np.int64))

input_size = X_train.shape[2]
ae_hidden_dim = 128  # Modify the hidden layer dimension of Autoencoder

print("Input feature dimension (input_size):", input_size)

# Create training dataset and loader
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model initialization and training
input_dim = feature_dim
num_classes = 12
model = DeepModel(input_dim, num_classes, ae_hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Model evaluation
model.eval()
with torch.no_grad():
    test_inputs = X_test.clone().detach().to(torch.float32)
    test_labels = y_test.clone().detach().to(torch.int64)
    test_outputs = model(test_inputs)
    test_loss = criterion(test_outputs, test_labels)
    _, predicted_labels = torch.max(test_outputs, 1)
    accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
    print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

# Save model
model_path = 'HIV-SPBEnv.pth'
torch.save(model.state_dict(), model_path)

















