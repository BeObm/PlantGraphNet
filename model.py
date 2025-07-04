from torch.nn.functional import relu, log_softmax
from torch_geometric.nn import global_add_pool
from utils import *
import torch
import torch.nn.init as init
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

class CNNModel(nn.Module):
    def __init__(self, num_classes=10,feature_siz=3):  # num_classes can be adjusted for your dataset
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



class HybridImageClassifier(nn.Module):
    def __init__(self, num_classes, feature_size):
        super(HybridImageClassifier, self).__init__()

        self.cnn =models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity() 
        
        # Freeze All Layers Initially
        for param in self.cnn.parameters():
            param.requires_grad = True  

        # Unfreeze Only Layer4 (last few layers for fine-tuning)
        for param in self.cnn.layer4.parameters():  
            param.requires_grad = True  
        cnn_output_size = 2048  

        # Additional Feature Processing
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 2048),  # Match ResNet output size
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2)
        )
        self._initialize_weights()
        # Fusion Layer (Combining CNN & Extracted Features)
        self.fusion_fc = nn.Sequential(
                nn.Linear(cnn_output_size, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)    

    def forward(self, image, additional_features):
        # Process image through CNN
        image_features = self.cnn(image)

        # Process additional extracted features
        extracted_features = self.feature_fc(additional_features)

        # Concatenate CNN features with additional extracted features
        fused_features= image_features*extracted_features

        # Pass through the final classifier
        output = self.fusion_fc(fused_features)

        return output






class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, Conv1, Conv2,image_feature=150, use_image_feats=False,f_model="unet"):
        super(GNNModel, self).__init__()
        set_seed()
        # Graph feature extraction layers
        self.graph_conv1 = Conv1(num_node_features[0], hidden_dim)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.act = torch.nn.ReLU()
        self.use_image_feats = use_image_feats
        self.f_model = f_model
        self.graph_conv2 = Conv2(hidden_dim, hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)

        self.graph_conv3 = Conv2(hidden_dim, hidden_dim)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_dim)
        
        self.node_feature_fc = torch.nn.Linear(hidden_dim, hidden_dim)  # Node feature transformation
        
        init.xavier_uniform_(self.node_feature_fc.weight)
        init.zeros_(self.node_feature_fc.bias)
        if self.use_image_feats==True:
            if self.f_model=="unet":
                self.image_feature_extractor = UNetFeatureExtractor(in_channels=3, feature_dim=image_feature)
                self.img_feature_fc = nn.Linear(512, hidden_dim)
            else:
            # Load a pretrained ResNet model for image feature extraction
                resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                self.image_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
                self.img_feature_fc = torch.nn.Linear(2048, hidden_dim)  # Map ResNet features to hidden_dim
            
            # Initialize weights for `img_feature_fc` layer
            init.xavier_uniform_(self.img_feature_fc.weight)
            init.zeros_(self.img_feature_fc.bias)
            self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)  # Classifier combining graph & image features
            
            init.xavier_uniform_(self.fc.weight)
            init.zeros_(self.fc.bias)
        else:
            self.fc = torch.nn.Linear(hidden_dim, num_classes)
            init.xavier_uniform_(self.fc.weight)
            init.zeros_(self.fc.bias)

        self.g_dropout = torch.nn.Dropout(p=0.0)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, data):
        # Graph feature processing
        node_features = data.x
        edge_index = data.edge_index.view(2, -1)
        edge_attr = data.edge_attr
        batch = data.batch
        
        node_features = self.graph_conv1(node_features, edge_index, edge_attr)
        node_features = self.batch_norm1(node_features)
        node_features = self.g_dropout(node_features)
        node_features = self.act(node_features)

        node_features = self.graph_conv2(node_features, edge_index, edge_attr)
        node_features = self.batch_norm2(node_features)
        node_features = self.g_dropout(node_features)
        node_features = self.act(node_features)

        node_features = self.graph_conv3(node_features, edge_index, edge_attr)
        node_features = self.batch_norm3(node_features)
        node_features = self.g_dropout(node_features)
        node_features = self.act(node_features)
        
        node_features = global_add_pool(node_features, batch)  # Global pooling for graph features
        node_features = self.node_feature_fc(node_features)

        # Image feature processing
        if self.use_image_feats==True:
            image_features = data.image_features  # Assuming images are already preprocessed to (batch, 3, H, W)
            if self.f_model=="unet":
                 image_feats = self.image_feature_extractor(image_features)  # (B, 512)
                 image_feats = self.img_feature_fc(image_feats)   
            else:
                img_features = self.image_feature_extractor(image_features)
                img_features = img_features.view(img_features.size(0), -1)  # Flatten
                img_features = self.img_feature_fc(img_features)
                
            combined_features = torch.cat([node_features, img_features], dim=1)  # Concatenate graph & image features
            combined_features = self.dropout(combined_features)
            output = self.fc(combined_features)
        else:
            output = self.fc(node_features)

        return output


class UNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=512):
        super(UNetFeatureExtractor, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (B, 512, 1, 1)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(self.pool1(x))
        x = self.encoder3(self.pool2(x))
        x = self.bottleneck(self.pool3(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Shape: (B, 512)
        return x  # Return feature vector



def train_function(epochs, model, train_dataloader,val_dataloader,type_graph, criterion, optimizer,scheduler,accelerator):
    pbar = tqdm(epochs)
    pbar.set_description("training model")
    train_losses = []
    train_accuracies = []
    best_loss=9999999999999999999

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            accelerator.backward(loss)
            optimizer.step()
            # scheduler.step()
            running_loss += accelerator.gather_for_metrics(loss).sum().item() * batch.num_graphs
        
        train_loss= running_loss / len(train_dataloader)
        train_losses.append(train_loss)
        if train_loss <= best_loss:
            best_loss = train_loss
            best_model=deepcopy(model)
        if epoch % 10 == 0:
            torch.save(best_model.state_dict(), f"results/GNN_Models/{type_graph}_best_model.pth")
        pbar.write(f'\n Epoch [{epoch}/{epochs}]: Loss: {round(train_loss, 5)} | Current best loss: {round(best_loss, 5)}')
        pbar.update(1)
    return train_losses,best_model


@torch.no_grad()
def test_function(accelerator, model, test_loader, class_names):
    model.eval()
    
    true_labels = []
    pred_labels = []
    
    filename = f"{config['param']['result_folder']}/confusion_matrix.pdf"
    
    for data in test_loader:  
        targets= data.y.to(accelerator.device)
        logits = model(data)
        pred= torch.argmax(logits,dim=1)
        
        all_targets =accelerator.gather_for_metrics(targets)
        all_pred = accelerator.gather_for_metrics(pred)
        
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
    


# @torch.no_grad()
# def test(model, loader,device,class_names):
#     filename = f"{config['param']['result_folder']}/confusion_matrix.pdf"
#     print(f"class name size is {len(class_names)}")
#     model.eval()
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for data in loader:
#             data.to(device)
#             out = model(data)
#             pred = out.argmax(dim=1)
#             y_true.extend(data.y.tolist())
#             y_pred.extend(pred.tolist())
            
   
#     plot_confusion_matrix(y_true=y_true,
#                           y_pred=y_pred,
#                           class_names=class_names,
#                           file_name=filename
#                           )
#     print(f"Confusion Matrix for the GNN model is saved in {filename}")

#     cls_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

#     return cls_report
