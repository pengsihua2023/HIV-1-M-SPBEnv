#Author: Sihua Peng, PhD
#Date： 08/20/2023

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from Bio import SeqIO

# Reading fasta format file
sequences = []
for record in SeqIO.parse('HIV_12-class-new.fasta', 'fasta'):
    sequences.append(str(record.seq))

# Defining k-mer size and feature dimension
k = 7
feature_dim = 4 ** k

# Creating feature vector
def create_feature_vector(sequence, k):
    feature_vector = np.zeros(4 ** k)
    for i in range(len(sequence)-k+1):
        kmer = sequence[i:i+k]
        index = kmer_to_index(kmer)
        feature_vector[index] += 1
    return feature_vector

# Converting k-mer to feature vector index
def kmer_to_index(kmer):
    index = 0
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, base in enumerate(kmer[::-1]):
        index += mapping[base] * (4 ** i)
    return index

# Creating feature matrix
feature_matrix = np.zeros((len(sequences), feature_dim))
for i, sequence in enumerate(sequences):
    feature_matrix[i] = create_feature_vector(sequence, k)

# Reading label file
labels = pd.read_csv('Labels_1068-12-class.csv')  # Assuming the label file is in CSV format

# Defining label mapping
label_mapping = {'A1': 0, 'A2': 1, 'B': 2, 'C': 3, 'D': 4, 'F1': 5, 'F2': 6, 'G': 7, 'H': 8, 'J': 9, 'K': 10, 'L': 11}

# Mapping labels to numbers
labels['Label'] = labels['Label'].map(label_mapping)

# Extracting feature and label data
X = feature_matrix
y = labels['Label'].values

# Defining neural network model
class DeepModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(512 * ((input_dim - 2) // 2), 128)  # Calculating input size for fully connected layer
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(torch.relu(x))
        x = x.view(x.size(0), -1)  # Flatten operation, flattening the feature map
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Data preparation
batch_size = 32  # Set appropriate batch size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train.astype(np.float32)).unsqueeze(1)
X_test = torch.tensor(X_test.astype(np.float32)).unsqueeze(1)
y_train = torch.tensor(y_train.astype(np.int64))
y_test = torch.tensor(y_test.astype(np.int64))

input_size = X_train.shape[2]

print("Input feature dimension (input_size):", input_size)

# Creating training dataset and loader
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Defining Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initializing Autoencoder model
autoencoder = Autoencoder(input_size)

# Model initialization and training
input_dim = feature_dim
num_classes = 12
model = DeepModel(input_dim, num_classes)
optimizer = optim.Adam(list(model.parameters()) + list(autoencoder.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    autoencoder.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        encoded_inputs = autoencoder(inputs.view(inputs.size(0), -1))
        outputs = model(encoded_inputs.view(inputs.size(0), 1, -1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Model evaluation
model.eval()
autoencoder.eval()
with torch.no_grad():
    test_inputs = X_test.clone().detach().to(torch.float32)
    encoded_test_inputs = autoencoder(test_inputs.view(test_inputs.size(0), -1))
    test_labels = y_test.clone().detach().to(torch.int64)
    test_outputs = model(encoded_test_inputs.view(test_inputs.size(0), 1, -1))
    test_loss = criterion(test_outputs, test_labels)
    _, predicted_labels = torch.max(test_outputs, 1)
    accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
    print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

# Save model
model_path = 'HIV-Deep-Autoencoder.pth'
torch.save(model.state_dict(), model_path)
























