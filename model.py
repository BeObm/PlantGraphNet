from torch.nn.functional import relu, log_softmax
from torch_geometric.nn import global_add_pool
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):  # num_classes can be adjusted for your dataset
        super(CNNModel, self).__init__()
        a=1024
        b=512
        c=256
        d=128
        e=64

        # 1 Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 2 Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=a, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(a)
        self.conv4 = nn.Conv2d(in_channels=a, out_channels=a, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(a)

        # 3 Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=a, out_channels=a, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(a)
        self.conv6 = nn.Conv2d(in_channels=a, out_channels=a, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(a)


        # 4 Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=a, out_channels=b, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(b)
        self.conv2 = nn.Conv2d(in_channels=b, out_channels=b, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(b)

        # 5 Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=b, out_channels=b, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(b)
        self.conv4 = nn.Conv2d(in_channels=b, out_channels=b, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(b)

        # 6 Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=b, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(c)
        self.conv6 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(c)


        # 7 Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c)

        # 8 Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=d, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(d)
        self.conv4 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(d)

        # 9 Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(d)


        # 10 Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=d, out_channels=e, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(e)
        self.conv2 = nn.Conv2d(in_channels=e, out_channels=e, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(e)

        # 11 Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=e, out_channels=e, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(e)
        self.conv4 = nn.Conv2d(in_channels=e, out_channels=e, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(e)

        # 12 Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=e, out_channels=a, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(a)
        self.conv6 = nn.Conv2d(in_channels=a, out_channels=a, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(a)


        # 13 Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=a, out_channels=b, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(b)
        self.conv2 = nn.Conv2d(in_channels=b, out_channels=b, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(b)

        # 14 Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=b, out_channels=b, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(b)
        self.conv4 = nn.Conv2d(in_channels=b, out_channels=b, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(b)

        # 15 Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=b, out_channels=s, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(c)
        self.conv6 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(c)

        # 16 Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c)

        # 17 Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=d, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(d)

        # 18 Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(d)
        self.conv6 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(d)

        # 19 Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=d, out_channels=e, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(e)
        self.conv4 = nn.Conv2d(in_channels=e, out_channels=e, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(e)

        # 20 Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=e, out_channels=e, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(e)
        self.conv6 = nn.Conv2d(in_channels=e, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)


        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 28 * 28, 64)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)

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
    # Original image input feature count

    def __init__(self, num_node_features, hidden_dim, num_classes, Conv1, Conv2,image_feature=50176,use_image_feats=False):
        super(GNNModel, self).__init__()

        # Graph feature extraction (upgraded to more advanced GNN layers like GAT or GIN)

        self.graph_conv1 = Conv1(num_node_features, hidden_dim * 2)
        self.batch_norm1= torch.nn.BatchNorm1d(hidden_dim*2)
        self.act=torch.nn.ReLU()
        self.use_image_feats=use_image_feats
        #
        self.graph_conv2 = Conv2(hidden_dim * 2, hidden_dim)
        self.batch_norm2= torch.nn.BatchNorm1d(hidden_dim)

        # Process image features with CNN (multiple convolution layers)
        self.img_cnn = torch.nn.Sequential(
            torch.nn.Linear(image_feature, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            # torch.nn.Linear(512, 256),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(256),
            # torch.nn.Linear(256, 128),  # Reduce to 128-dimensional feature vector
            # torch.nn.ReLU(),
            torch.nn.Linear(512, hidden_dim),  # Map final features to `hidden_dim`
        )

        # Linear layers for combining graph features and image features
        self.img_feature_fc = torch.nn.Linear(64, hidden_dim)  # Map image features to hidden dim
        self.node_feature_fc = torch.nn.Linear(hidden_dim, hidden_dim)  # Node feature transformation
        if self.use_image_feats:
           self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)  # Classifier
        else:
            self.fc=torch.nn.Linear(hidden_dim,num_classes)

        # Dropout to prevent overfitting
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        # Graph feature processing
        node_features = data.x
        edge_index = data.edge_index.view(2, -1)
        batch = data.batch
        image_features=data.image_features
        # print(f"node_features: {node_features.shape} edge_index: {edge_index.shape} batch: {batch.shape} image_features: {image_features.shape}")
        node_features = self.graph_conv1(node_features, edge_index)
        node_features = self.batch_norm1(node_features)
        node_features = self.act(node_features)

        node_features = self.graph_conv2(node_features, edge_index)
        node_features = self.batch_norm2(node_features)
        node_features = self.act(node_features)

        node_features = global_add_pool(node_features, batch)  # Global pooling for graph features
        node_features = self.node_feature_fc(node_features)

        # Image feature processing
        if self.use_image_feats:
            img_features = image_features  # Assuming it's a 4D tensor for CNN ([batch, C, H, W])
            img_features = self.img_cnn(img_features).view(img_features.size(0), -1)  # Flatten after pooling
            img_features = self.img_feature_fc(img_features)
            combined_features = torch.cat([node_features, img_features], dim=1)  # Concatenate graph & image features
            combined_features = self.dropout(combined_features)  # Apply dropout
            output = self.fc(combined_features)  # Classify
        else:
            output = self.fc(node_features)  # Classify


        # Effective fusion of graph and image features



        return log_softmax(output, dim=1)





class GNNModel0(torch.nn.Module):
    def __init__(self, num_node_features,hidden_dim, num_classes,Conv1,Conv2):
        super(GNNModel, self).__init__()
        self.conv1 = Conv1(num_node_features, hidden_dim*2)
        self.conv2 = Conv2(hidden_dim*2, hidden_dim)
        self.mlp_x0 = torch.nn.Linear(3, hidden_dim)

        self.mlp_x = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)


    def forward(self, data):
        x, edge_index, img_feature,batch = data.x, data.edge_index.view(2,-1), data.image_features, data.batch

        x0 = self.mlp_x0(img_feature)
        x = relu(self.conv1(x, edge_index))
        x = relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)  # Global add pooling
        x = self.mlp_x(x)

        print(f" Before cat| x0: {x0.shape} x: {x.shape}")
        # xt = torch.cat([x0,x],1)
        # print(f" After cat |x0: {x0.shape} x: {x.shape}, xt: {xt.shape}")

        x = self.fc(x)

        return log_softmax(x, dim=1)


def train(model, train_loader, optimizer, criterion, device):
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

def test(model, loader,device,class_names):
    filename = f"{config['param']['result_folder']}/confusion_matrix.pdf"

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

    plot_confusion_matrix(y_true=y_true,
                          y_pred=y_pred,
                          class_names=class_names,
                          file_name=filename
                          )
    print(f"Confusion Matrix for the GNN model is saved in {filename}")

    cls_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    return cls_report