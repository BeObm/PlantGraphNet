import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from model import CNNModel
from utils import *
import os
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", torch.cuda.is_available())
# device = torch.device('cpu')

set_seed()
print("Loading dataset...")
num_classes, train_loader = load_data(dataset_dir="dataset/images/train", batch_size=32)
_, test_loader = load_data(dataset_dir="dataset/images/val", batch_size=32)

model = CNNModel()


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
num_epochs =100
best_validation_accuracy = 0.0
best_model_state = model.state_dict()
# Lists to store loss and accuracy values for plotting
train_loss_values = []
validation_loss_values = []
train_accuracy_values = []
validation_accuracy_values = []
print("Trainning Model...")

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



# Testing
# Testing phase
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.append(predicted.cpu())  # Append tensors to list after moving to CPU
        y_true.append(labels.cpu())     # Append tensors to list after moving to CPU

# Concatenate the list of tensors into a single tensor for y_pred and y_true
y_pred = torch.cat(y_pred)
y_true = torch.cat(y_true)

# If compute_metrics function accepts PyTorch tensors, then you can directly pass them
metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
for metric, value in metrics.items():
    print(f"{metric} = {value}")

with open("baseline_result.txt", 'a') as baseline:
    baseline.write(f"{'+'*12}Model = AlexNet {'+'*12}\n")
    for metric, value in metrics.items():
        baseline.write(f"{metric} = {value}")
    baseline.write("=="*25 + "\n")
