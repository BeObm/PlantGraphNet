import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from utils import *
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader, random_split

set_seed()

print("Loading dataset...")
num_classes, train_loader, validation_loader, test_loader = load_data(dataset_dir="../dataset/images/Corn0", batch_size=32)

model = models.vgg19(pretrained='vggface', progress=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if a saved model exists and load it
saved_model_path = 'best_model.pth'

if os.path.isfile(saved_model_path):
    try:
        model.load_state_dict(torch.load(saved_model_path))
        print(f"Loaded saved model from {saved_model_path}")
    except:
        pass

# Training loop
num_epochs = 100
best_validation_accuracy = 0.0
best_model_state = model.state_dict()
# Lists to store loss and accuracy values for plotting
train_loss_values = []
validation_loss_values = []
train_accuracy_values = []
validation_accuracy_values = []
print("Training Model...")

for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Compute and store training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_loss_values.append(train_loss)
    train_accuracy = 0
    total = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    train_accuracy_values.append(train_accuracy)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%')

    # Validation
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total
    validation_accuracy_values.append(validation_accuracy)
    print(f'Validation Accuracy: {validation_accuracy}%')

    # Save the model if it's the best so far
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_model_state = model.state_dict()

# Save the best model to disk
torch.save(best_model_state, 'best_model.pth')

# Load the best model for testing
model.load_state_dict(best_model_state)

# Testing
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Compute metrics
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred, average='weighted'),
    'recall': recall_score(y_true, y_pred, average='weighted'),
    'f1': f1_score(y_true, y_pred, average='weighted')
}

# Print and write metrics to file
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value}")

with open("baseline_result.txt", 'a') as baseline:
    baseline.write(f"{'+'*12} Model = vgg19 {'+'*12}\n")
    for metric, value in metrics.items():
        baseline.write(f"{metric.capitalize()}: {value}\n")
    baseline.write("="*50 + "\n")
