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


class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, Conv1, Conv2,
                 image_feature=150, use_image_feats=False, image_backbone='resnet'):
        super(GNNModel, self).__init__()
        set_seed()

        # Graph layers
        self.graph_conv1 = Conv1(num_node_features[0], hidden_dim)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.graph_conv2 = Conv2(hidden_dim, hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)
        self.graph_conv3 = Conv2(hidden_dim, hidden_dim)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_dim)
        self.act = torch.nn.ReLU()
        self.node_feature_fc = torch.nn.Linear(hidden_dim, hidden_dim)
        init.xavier_uniform_(self.node_feature_fc.weight)
        init.zeros_(self.node_feature_fc.bias)

        self.use_image_feats = use_image_feats

        if self.use_image_feats:
            self.image_feature_extractor = baseline_model(model_name=image_backbone, num_classes=hidden_dim)
            # self.image_feature_extractor= self.image_feature_extractor[:-1]
            img_feature_dim=2048
            # if self.image_backbone == 'resnet':
            #     resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            #     self.image_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove FC
            #     img_feature_dim = 2048
            # elif self.image_backbone == 'yolo8':
            #     yolov8 = YOLO("yolov8n.pt")  # or yolov8s.pt/m.pt etc.
            #     self.image_feature_extractor = yolov8.model.model[:20]  # up to backbone
            #     img_feature_dim = 512  # depends on the model; adjust if using larger models
            # else:
            #     raise ValueError(f"Unsupported image backbone: {self.image_backbone}")

            self.img_feature_fc = torch.nn.Linear(hidden_dim, hidden_dim)
            init.xavier_uniform_(self.img_feature_fc.weight)
            init.zeros_(self.img_feature_fc.bias)

            self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)
            init.xavier_uniform_(self.fc.weight)
            init.zeros_(self.fc.bias)
        else:
            self.fc = torch.nn.Linear(hidden_dim, num_classes)
            init.xavier_uniform_(self.fc.weight)
            init.zeros_(self.fc.bias)

        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, data):
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

        node_features = global_add_pool(node_features, batch)
        node_features = self.node_feature_fc(node_features)

        if self.use_image_feats:
            image_features = data.image_features  # (B, 3, H, W)
            img_feats = self.image_feature_extractor(image_features)

            # if self.image_backbone == 'resnet':
            #     img_feats = img_feats.view(img_feats.size(0), -1)  # (B, 2048)
            # elif self.image_backbone == 'yolo':
            #     if isinstance(img_feats, list):  # YOLOv8 returns list of feature maps
            #         img_feats = img_feats[-1]
            #     img_feats = F.adaptive_avg_pool2d(img_feats, (1, 1))
            #     img_feats = img_feats.view(img_feats.size(0), -1)  # (B, 512)

            img_feats = self.img_feature_fc(img_feats)
            combined = torch.cat([node_features, img_feats], dim=1)
            combined = self.dropout(combined)
            output = self.fc(combined)
        else:
            output = self.fc(node_features)
        return output





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
