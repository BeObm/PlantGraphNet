import pandas as pd
from torch.nn.functional import relu, log_softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn import global_add_pool

from utils import *


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
