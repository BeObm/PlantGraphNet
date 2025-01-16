import pandas as pd
from torch.nn.functional import relu, log_softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn import global_add_pool
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):  # num_classes can be adjusted for your dataset
        super(CNNModel, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Second Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Third Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First Convolutional Block with ReLU and Max Pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Second Convolutional Block with ReLU and Max Pooling
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Third Convolutional Block with ReLU and Max Pooling
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Flatten the feature map
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer

        # Fully Connected Layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x



class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features,hidden_dim, num_classes,Conv1,Conv2):
        super(GNNModel, self).__init__()
        self.conv1 = Conv1(num_node_features, hidden_dim*2)
        self.conv2 = Conv2(hidden_dim*2, hidden_dim)
        self.mlp_x0 = torch.nn.Linear(50176, 1024)
        self.mlp_x1 = torch.nn.Linear(1024, hidden_dim)

        self.mlp_x = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim*2, num_classes)


    def forward(self, data):
        x, edge_index, img_feature,batch = data.x, data.edge_index, data.img_features, data.batch

        x0 = self.mlp_x0(img_feature)
        x0 = self.mlp_x1(x0)
        x = relu(self.conv1(x, edge_index))
        x = relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)  # Global add pooling
        x = self.mlp_x(x)

        xt = torch.cat((x0,x),1)

        x = self.fc(xt)

        return log_softmax(x, dim=1)


def train(model, train_loader, optimizer, criterion, device=device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(data.y).sum().item()
        total += data.y.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def test(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.tolist())
            y_pred.extend(pred.tolist())

    dict_pre={"y_true":y_true,"y_pred":y_pred}
    dp=pd.DataFrame(dict_pre)
    dp.to_csv("test_pred.csv")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
