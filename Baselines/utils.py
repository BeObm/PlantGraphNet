from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torchvision import datasets, transforms
from torch.utils.data import  SubsetRandomSampler,DataLoader
import numpy as np
import os
import random
from collections import defaultdict



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


def load_data(dataset_dir, batch_size=16, n=200):
    # Create datasets
    dataset = datasets.ImageFolder(dataset_dir, transform=transform())
    num_classes = len(dataset.classes)
    targets = [dataset.targets[i] for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    # Create data loaders
    num_samples_per_class = n

    sampler = BalancedSampler(
        indices=indices,
        num_samples_per_class=num_samples_per_class,
        class_to_idx=dataset.class_to_idx,
        targets=targets
    )

    # total_length = len(dataset)
    # train_size = int(0.8 * total_length)
    # val_size = int(0.10 * total_length)
    # test_size = total_length - train_size - val_size
    # train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)  # shuffle=True,
    print(f"Dataset details: {count_classes(data_loader)}")

    # validation_loader = DataLoader(val_set, batch_size=batch_size)
    # test_loader = DataLoader(test_set, batch_size=batch_size)

    return num_classes, data_loader, dataset.classes


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


class BalancedSampler(SubsetRandomSampler):
    def __init__(self, indices, num_samples_per_class, class_to_idx, targets):
        self.indices = indices
        self.num_samples_per_class = num_samples_per_class
        self.class_to_idx = class_to_idx
        self.targets = targets
        self.balanced_indices = self.get_balanced_indices()
        super().__init__(self.balanced_indices)

    def get_balanced_indices(self):
        # Group indices by class
        class_indices = defaultdict(list)
        for idx in self.indices:
            class_label = self.targets[idx]
            class_indices[class_label].append(idx)

        # Sample equal number of images per class
        balanced_indices = []
        for class_label, indices in class_indices.items():
            balanced_indices.extend(
                random.sample(indices, min(self.num_samples_per_class, len(indices)))
            )

        return balanced_indices


def count_classes(dataloader):
    class_counts = defaultdict(int)
    for _, labels in dataloader:
        for label in labels:
            class_counts[label.item()] += 1



    return class_counts