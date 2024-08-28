import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
torch.autograd.set_detect_anomaly(True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
def kmer_to_index(kmer, k):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    index = 0
    for i, base in enumerate(kmer[::-1]):
        index += mapping[base] * (4 ** i)
    return index

def create_feature_vector(sequence, k):
    feature_vector = np.zeros(4 ** k)
    for i in range(len(sequence)-k+1):
        kmer = sequence[i:i+k]
        index = kmer_to_index(kmer, k)
        feature_vector[index] += 1
    return feature_vector

sequences = []
for record in SeqIO.parse('./data/HIV_12-class.fasta', 'fasta'):
    sequences.append(str(record.seq))

k = 6 
feature_dim = 4 ** k

feature_matrix = np.zeros((len(sequences), feature_dim))
for i, sequence in enumerate(sequences):
    feature_matrix[i] = create_feature_vector(sequence, k)

labels = pd.read_csv('./data/Labels-12-class.csv')
label_mapping = {'A1': 0, 'A2': 1, 'B': 2, 'C': 3, 'D': 4, 'F1': 5, 'F2': 6, 'G': 7, 'H': 8, 'J': 9, 'K': 10, 'L': 11}
labels['Label'] = labels['Label'].map(label_mapping)
y = labels['Label'].values

X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train.astype(np.float32)).to(device)
X_test = torch.tensor(X_test.astype(np.float32)).to(device)
y_train = torch.tensor(y_train.astype(np.int64)).to(device)
y_test = torch.tensor(y_test.astype(np.int64)).to(device)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.unsqueeze(1)
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = out + identity
        out = self.relu2(out)
        return out

class TransposeResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(TransposeResidualBlock, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, padding=1, output_padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, padding=1, output_padding=0, bias=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = out + identity
        out = self.relu2(out)
        return out


# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, ae_hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            # 这里去掉了最后一个 ResidualBlock(128)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            TransposeResidualBlock(64),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            TransposeResidualBlock(32),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x



class DeepModel(nn.Module):
    def __init__(self, input_dim, num_classes, ae_hidden_dim):
        super(DeepModel, self).__init__()
        self.autoencoder = Autoencoder(input_dim, ae_hidden_dim)
        self.fc1 = nn.Linear(4**(k+2), 128)  # Adjusting input feature count
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        encoded_x, decoded_x = self.autoencoder(x)
        encoded_x = encoded_x.view(encoded_x.size(0), -1)
        x = torch.relu(self.fc1(encoded_x))
        x = torch.relu(self.fc2(x))
        classification_output = self.fc3(x)
        return encoded_x, decoded_x, classification_output

# Initialize and train the model
model = DeepModel(feature_dim, len(label_mapping), 128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

print (model)

# Training and validation logic remains the same

# Training and validation code
num_epochs = 20
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_reconstruction_loss = 0
    total_classification_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        _, decoded_x, classification_outputs = model(inputs)
        reconstruction_loss = mse_loss(decoded_x, inputs)
        classification_loss = criterion(classification_outputs, labels)
        loss = reconstruction_loss + classification_loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_classification_loss += classification_loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, decoded_x, classification_outputs = model(inputs)
            reconstruction_loss = mse_loss(decoded_x, inputs)
            classification_loss = criterion(classification_outputs, labels)
            loss = reconstruction_loss + classification_loss
            total_test_loss += loss.item()
            _, predicted = torch.max(classification_outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    test_accuracy = total_correct / len(test_loader.dataset)
    test_accuracies.append(test_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Total Train Loss: {avg_train_loss:.4f}, Total Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


# Plotting results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, color='green', label='Total Train Loss', linewidth=2)
plt.plot(test_losses, color='red', label='Total Test Loss', linewidth=3)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('Total Train vs Total Test Loss', fontsize=18)
plt.legend(fontsize=16)

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, color='blue', label='Test Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.title('Test Accuracy', fontsize=18)
plt.legend(fontsize=18)
plt.show()

# Save model
model_path = 'model-HIV_1_M_SPBEnv-12.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Evaluation on an independent validation set
validation_sequences = []
for record in SeqIO.parse('./data/validation_data.fasta', 'fasta'):
    validation_sequences.append(str(record.seq))

validation_feature_matrix = np.zeros((len(validation_sequences), feature_dim))
for i, sequence in enumerate(validation_sequences):
    validation_feature_matrix[i] = create_feature_vector(sequence, k)

validation_labels = pd.read_csv('./data/validation_Label.csv', header=None)
validation_labels = validation_labels.iloc[:, 0].map(label_mapping).values

validation_features = torch.tensor(validation_feature_matrix.astype(np.float32)).to(device)
validation_labels = torch.tensor(validation_labels.astype(np.int64)).to(device)

model.eval()
validation_outputs = []
true_labels = []
with torch.no_grad():
    for i in range(0, len(validation_features), 32):
        inputs = validation_features[i:i+32].unsqueeze(1)
        labels = validation_labels[i:i+32]
        _, _, classification_outputs = model(inputs)
        _, predicted = torch.max(classification_outputs.data, 1)
        validation_outputs.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate and display evaluation metrics
accuracy = accuracy_score(true_labels, validation_outputs)
recall = recall_score(true_labels, validation_outputs, average='macro')
precision = precision_score(true_labels, validation_outputs, average='macro')
f1 = f1_score(true_labels, validation_outputs, average='macro')

print(f"Validation Accuracy: {accuracy}")
print(f"Validation Recall: {recall}")
print(f"Validation Precision: {precision}")
print(f"Validation F1 Score: {f1}")

# Confusion matrix
class_names = ['A1', 'A2', 'B', 'C', 'D', 'F1', 'F2', 'G', 'H', 'J', 'K', 'L']
cm = confusion_matrix(true_labels, validation_outputs)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_xlabel('Predicted label', fontsize=12)
ax.set_ylabel('True label', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.xticks(rotation=45)
plt.show()

