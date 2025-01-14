from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
set_seed()

print("Loading dataset...")
num_classes, train_loader = load_data(dataset_dir="../dataset/images/train", batch_size=64)
_, test_loader = load_data(dataset_dir="../dataset/images/val", batch_size=64)
# Load a pre-trained ResNet-50 model
model = models.vgg16(pretrained='vggface', progress=True)

for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

# Modify the classifier for your classification task

model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

# Move the model to the GPU
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

    # with torch.no_grad():
    #     for inputs, labels in validation_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # validation_accuracy = 100 * correct / total
    # validation_accuracy_values.append(validation_accuracy)
    # print(f'Validation Accuracy: {validation_accuracy}%')
    #
    # # Save the model if it's the best so far
    # if validation_accuracy > best_validation_accuracy:
    #     best_validation_accuracy = validation_accuracy
    #     best_model_state = model.state_dict()

# Save the best model to disk
torch.save(best_model_state, 'best_model.pth')

# Load the best model for testing
model.load_state_dict(best_model_state)

# Testing
model.eval()
correct = 0
total = 0
with torch.no_grad():
    y_pred=[]
    y_true=[]
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.append(predicted)
        y_true.append(labels)

y_pred = torch.cat(y_pred).cpu().numpy()
y_true = torch.cat(y_true).cpu().numpy()

metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
for metric, value in metrics.items():
    print(f"{metric} = {value}")

with open("baseline_result.txt", 'a') as baseline:
    baseline.write(f"{'+'*12}Model = vgg16 {'+'*12}\n")
    for metric, value in metrics.items():
        baseline.write(f"{metric} = {value}")
    baseline.write("=="*25 + "\n")