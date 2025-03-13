from torch.nn.functional import relu, log_softmax
from torch_geometric.nn import global_add_pool
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.cuda.amp import autocast, GradScaler
from torchvision import models


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):  # num_classes can be adjusted for your dataset
        super(CNNModel, self).__init__()
        
        a0=2048
        a=1024
        b=512
        c=256
        d=128
        e=64
        f=32
        g=16

        # 1 Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=a0, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(a0)
        self.conv2 = nn.Conv2d(in_channels=a0, out_channels=a0, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(a0)

        # 2 Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=a0, out_channels=a, kernel_size=3, stride=1, padding=1)
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
        self.conv5 = nn.Conv2d(in_channels=e, out_channels=f, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(f)
        self.conv6 = nn.Conv2d(in_channels=f, out_channels=f, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(f)


        # 13 Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=f, out_channels=f, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(f)
        self.conv2 = nn.Conv2d(in_channels=f, out_channels=f, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(f)

        # 14 Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=f, out_channels=g, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(g)
        self.conv4 = nn.Conv2d(in_channels=g, out_channels=g, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(g)

        # 15 Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=g, out_channels=g, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(g)
        self.conv6 = nn.Conv2d(in_channels=g, out_channels=g, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(g)

        # # 16 Convolutional Block
        # self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(c)
        # self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(c)

        # # 17 Convolutional Block
        # self.conv3 = nn.Conv2d(in_channels=c, out_channels=d, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.conv4 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(d)

        # # 18 Convolutional Block
        # self.conv5 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(d)
        # self.conv6 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, stride=1, padding=1)
        # self.bn6 = nn.BatchNorm2d(d)

        # # 19 Convolutional Block
        # self.conv3 = nn.Conv2d(in_channels=d, out_channels=e, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(e)
        # self.conv4 = nn.Conv2d(in_channels=e, out_channels=e, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(e)

        # # 20 Convolutional Block
        # self.conv5 = nn.Conv2d(in_channels=e, out_channels=e, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(e)
        # self.conv6 = nn.Conv2d(in_channels=e, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.bn6 = nn.BatchNorm2d(256)


        # Fully Connected Layers
        self.fc1 = nn.Linear(g * 28 * 28, num_classes)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(num_classes, num_classes)
        self.fc3 = nn.Linear(num_classes, num_classes)

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





class GNNModel0(torch.nn.Module):
    # Original image input feature count

    def __init__(self, num_node_features, hidden_dim, num_classes, Conv1, Conv2,image_feature=150,use_image_feats=False):
        super(GNNModel, self).__init__()

        # Graph feature extraction (upgraded to more advanced GNN layers like GAT or GIN)

        self.graph_conv1 = Conv1(num_node_features[0], hidden_dim )
        self.batch_norm1= torch.nn.BatchNorm1d(hidden_dim)
        self.act=torch.nn.ReLU()
        self.use_image_feats=use_image_feats
            
        #
        self.graph_conv2 = Conv2(hidden_dim, hidden_dim)
        self.batch_norm2= torch.nn.BatchNorm1d(hidden_dim)

        self.graph_conv3 = Conv2(hidden_dim , hidden_dim)
        self.batch_norm2= torch.nn.BatchNorm1d(hidden_dim)
      
        # Linear layers for combining graph features and image features
        self.node_feature_fc = torch.nn.Linear(hidden_dim, hidden_dim)  # Node feature transformation
        if self.use_image_feats==True:
              # Process image features with CNN (multiple convolution layers)
            self.img_cnn = torch.nn.Sequential(
            torch.nn.Linear(image_feature, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, 128),  # Reduce to 128-dimensional feature vector
            torch.nn.ReLU(),
            torch.nn.Linear(128, hidden_dim),  # Map final features to `hidden_dim`
            )
            self.img_feature_fc = torch.nn.Linear(hidden_dim, hidden_dim)  # Map image features to hidden dim
            self.img_feature_fc = torch.nn.Linear(hidden_dim, hidden_dim)  # Map image features to hidden dim


            self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)  # Classifier
        else:
            self.fc=torch.nn.Linear(hidden_dim,num_classes)

        # Dropout to prevent overfitting
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        # Graph feature processing
        node_features = data.x
        edge_index = data.edge_index.view(2, -1)
        edge_attr = data.edge_attr
        batch = data.batch
            
        # print(f"node_features: {node_features.shape} edge_index: {edge_index.shape} batch: {batch.shape} image_features: {image_features.shape}")
        node_features = self.graph_conv1(node_features, edge_index,edge_attr)
        node_features = self.batch_norm1(node_features)
        node_features = self.act(node_features)

        node_features = self.graph_conv2(node_features, edge_index,edge_attr)
        node_features = self.batch_norm2(node_features)
        node_features = self.act(node_features)

        node_features = self.graph_conv3(node_features, edge_index,edge_attr)
        node_features = self.batch_norm2(node_features)
        node_features = self.act(node_features)
        
        node_features = global_add_pool(node_features, batch)  # Global pooling for graph features
        node_features = self.node_feature_fc(node_features)

        # Image feature processing
        if self.use_image_feats:
            image_features=data.image_features.view(data.image_features.size(0), -1)
            img_features = image_features  # Assuming it's a 4D tensor for CNN ([batch, C, H, W])
            img_features = self.img_cnn(img_features)  # Flatten after pooling
            img_features = self.img_feature_fc(img_features)
            combined_features = torch.cat([node_features, img_features], dim=1)  # Concatenate graph & image features
            combined_features = self.dropout(combined_features)  # Apply dropout
            output = self.fc(combined_features)  # Classify
        else:
            output = self.fc(node_features)  # Classify


        # Effective fusion of graph and image features



        return log_softmax(output, dim=1)




class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, Conv1, Conv2,image_feature=150, use_image_feats=False):
        super(GNNModel, self).__init__()

        # Graph feature extraction layers
        self.graph_conv1 = Conv1(num_node_features[0], hidden_dim)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.act = torch.nn.ReLU()
        self.use_image_feats = use_image_feats
        
        self.graph_conv2 = Conv2(hidden_dim, hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)

        self.graph_conv3 = Conv2(hidden_dim, hidden_dim)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_dim)
        
        self.node_feature_fc = torch.nn.Linear(hidden_dim, hidden_dim)  # Node feature transformation

        if self.use_image_feats:
            # Load a pretrained ResNet model for image feature extraction
            resnet = models.resnet50(pretrained=True)
            self.image_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
            self.img_feature_fc = torch.nn.Linear(2048, hidden_dim)  # Map ResNet features to hidden_dim
            
            self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)  # Classifier combining graph & image features
        else:
            self.fc = torch.nn.Linear(hidden_dim, num_classes)

        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, data):
        # Graph feature processing
        node_features = data.x
        edge_index = data.edge_index.view(2, -1)
        edge_attr = data.edge_attr
        batch = data.batch
        
        node_features = self.graph_conv1(node_features, edge_index, edge_attr)
        node_features = self.batch_norm1(node_features)
        node_features = self.act(node_features)

        node_features = self.graph_conv2(node_features, edge_index, edge_attr)
        node_features = self.batch_norm2(node_features)
        node_features = self.act(node_features)

        node_features = self.graph_conv3(node_features, edge_index, edge_attr)
        node_features = self.batch_norm3(node_features)
        node_features = self.act(node_features)
        
        node_features = global_add_pool(node_features, batch)  # Global pooling for graph features
        node_features = self.node_feature_fc(node_features)

        # Image feature processing
        if self.use_image_feats:
            image_features = data.image_features  # Assuming images are already preprocessed to (batch, 3, H, W)
            img_features = self.image_feature_extractor(image_features)
            img_features = img_features.view(img_features.size(0), -1)  # Flatten
            img_features = self.img_feature_fc(img_features)
            
            combined_features = torch.cat([node_features, img_features], dim=1)  # Concatenate graph & image features
            combined_features = self.dropout(combined_features)
            output = self.fc(combined_features)
        else:
            output = self.fc(node_features)

        return F.log_softmax(output, dim=1)







def train_function(model, dataloader, criterion, optimizer,accelerator):
    model.train()
    loss_all = 0
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        accelerator.backward(loss)
        loss_all += batch.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataloader) 


@torch.no_grad()
def test_function(accelerator, model, test_loader, class_names):
    model.eval()
    true_labels = []
    pred_labels = []
    filename = f"{config['param']['result_folder']}/confusion_matrix.pdf"
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        targets= data.y
        pred = model(data).max(dim=1)[1]
        all_targets =accelerator.gather(targets)
        all_pred = accelerator.gather(pred)
        true_labels.extend(all_targets.detach().cpu().numpy())
        pred_labels.extend(all_pred.detach().cpu().numpy())
    
    plot_confusion_matrix(y_true=true_labels,
                          y_pred=pred_labels,
                          class_names=class_names,
                          file_name=filename
                          )
    print(f"Confusion Matrix for the GNN model is saved in {filename}")

    cls_report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)

    return cls_report    
    


@torch.no_grad()
def test(model, loader,device,class_names):
    filename = f"{config['param']['result_folder']}/confusion_matrix.pdf"
    print(f"class name size is {len(class_names)}")
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

@torch.no_grad()
def test_ddp(model, loader,class_names):
    filename = f"{config['param']['result_folder']}/confusion_matrix.pdf"
    print(f"class name size is {len(class_names)}")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
           
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