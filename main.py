import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import matplotlib.pyplot as plt


# Check GPU support and set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Read fasta format file
sequences = []
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
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, dilation=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, dilation=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=8)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.attention = SelfAttention(embed_size=ae_hidden_dim // 8, heads=8)
        self.fc1 = nn.Linear(ae_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.autoencoder = Autoencoder(128 * ((input_dim - 2 * 2 - 4 * 2 - 8 * 2) // 2), ae_hidden_dim)  # Adjusted for dilation

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(torch.relu(x))
        x = self.conv3(torch.relu(x))
        x = self.pool(torch.relu(x))
        x = x.view(x.size(0), -1)
        encoded_x = self.autoencoder(x)
        
        reshaped_x = encoded_x.view(encoded_x.shape[0], 8, -1)
        attention_out = self.attention(reshaped_x, reshaped_x, reshaped_x)
        attention_out = attention_out.view(attention_out.shape[0], -1)
        
        x = torch.relu(self.fc1(attention_out))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Custom data set class
# Custom data set class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# data preparation
# Data preparation
batch_size = 32
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train.astype(np.float32)).unsqueeze(1).to(device)
X_test = torch.tensor(X_test.astype(np.float32)).unsqueeze(1).to(device)
y_train = torch.tensor(y_train.astype(np.int64)).to(device)
y_test = torch.tensor(y_test.astype(np.int64)).to(device)

input_size = X_train.shape[2]
ae_hidden_dim = 128  

print("Input feature dimension (input_size):", input_size)

# Create training dataset and loader
# Create training and validation dataset and loader
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Model initialization and training
input_dim = feature_dim
num_classes = 12
model = DeepModel(input_dim, num_classes, ae_hidden_dim).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 30
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_test_loss = 0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

    train_loss = total_train_loss / len(train_loader)
    test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_correct / total

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Draw a loss curve
plt.plot(range(num_epochs), train_losses, color='green', label='Train Loss')
plt.plot(range(num_epochs), test_losses, color='red', label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.legend()
plt.show()

# Save model
model_path = 'HIV-SPBEnv-kd-GPU7-7.pth'
torch.save(model.state_dict(), model_path)
