from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
from torchvision import datasets, transforms
from torch.utils.data import  random_split
from torch.utils.data.dataloader import DataLoader
import numpy as np
import os
import random



def compute_metrics(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}



def transform():
        return transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def load_data(dataset_dir,batch_size=16):
        # Create datasets
        dataset = datasets.ImageFolder(dataset_dir, transform=transform())
        num_classes = len(dataset.classes)
        # Create data loaders

        # total_length = len(dataset)
        # train_size = int(0.8 * total_length)
        # val_size = int(0.10 * total_length)
        # test_size = total_length - train_size - val_size
        # train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])


        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # validation_loader = DataLoader(val_set, batch_size=batch_size)
        # test_loader = DataLoader(test_set, batch_size=batch_size)

        return num_classes,train_loader

def set_seed():
    # os.CUBLAS_WORKSPACE_CONFIG="4096:8"
    seed=42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False