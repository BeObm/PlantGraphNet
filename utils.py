from collections import defaultdict
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from collections import Counter
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import sobel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv
import os.path as osp
from configparser import ConfigParser
from datetime import datetime
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from openpyxl import load_workbook
from skimage import io
from sklearn.metrics import confusion_matrix
from torchvision import datasets
from torch_geometric.loader import DataLoader, ImbalancedSampler
import concurrent.futures
from torch.utils.data import  SubsetRandomSampler,WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader as image_DataLoader
import random
import torch.nn.functional as F
import torch.nn as nn
import cv2
from tqdm import tqdm
from datetime import datetime
from skimage.filters import sobel
import networkx as nx
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern
from skimage.future import graph
from skimage.measure import regionprops
from skimage.util import img_as_float
from skimage.filters import sobel
from skimage.feature import hog, ORB
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50

import shutil

config = ConfigParser()
RunCode = dates = datetime.now().strftime("%d-%m_%Hh%M")
project_root_dir = os.path.abspath(os.getcwd())
fixe_size=50

def create_config_file(type_graph):
    configs_folder = osp.join(project_root_dir, f'results/GNN_Models/{type_graph}/{RunCode}')
    os.makedirs(configs_folder, exist_ok=True)
    config_filename = f"{configs_folder}/ConfigFile_{RunCode}.ini"
    graph_filename = f"{project_root_dir}/dataset/graphs"
    os.makedirs(graph_filename, exist_ok=True)
    config["param"] = {
        'config_filename': config_filename,
        'type_graph': type_graph,
        "graph_filename":graph_filename,
        "train_image_dataset_root": f"{project_root_dir}/dataset/images/train",
        "val_image_dataset_root": f"{project_root_dir}/dataset/images/val",
        "test_image_dataset_root": f"{project_root_dir}/dataset/images/test",

        "graph_dataset_folder": f"{graph_filename}",
        "result_folder": f"{configs_folder}",
        "sigma":1.0,
        "threshold":0.01,
        "max_corners":500,
        "grid_size": 10
    }

    with open(config_filename, "w") as file:
        config.write(file)


def plot_image_with_nodes(img_path, data, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the image
    img = io.imread(img_path)

    # Initialize the graph
    G = nx.Graph()

    for i, feat in enumerate(data.x.cpu().numpy()):
        x=feat[0]
        y=feat[1]
        G.add_node(i, pos=(x, y))

    for edge in data.edge_index.t().cpu().numpy():
        G.add_edge(edge[0], edge[1])

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(122)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=False, node_size=10, node_color='r')
    plt.axis('off')

    # Extract the image name without extension and create the output path
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_folder, img_name + '_graph.png')

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def save_plots(train_losses, metrics_dict):
    plt.figure(figsize=(12, 4))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Metrics
    plt.subplot(1, 2, 2)
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())
    plt.bar(metrics_names, metrics_values)
    plt.title('Test Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')

    plt.tight_layout()
    plt.savefig(f'{config["param"]["result_folder"]}/training_plots.png')
    plt.show()

    # Save Training Evolution Data to CSV
    training_data = {'Iteration': list(range(1, len(train_losses) + 1)), 'Training Loss': train_losses}
    training_df = pd.DataFrame(training_data)
    training_df.to_csv(f"{config['param']['result_folder']}/training_evolution.csv", index=False)


def plot_and_save_training_performance(num_epochs, losses, folder_name):
    csv_file=f"{folder_name}/training_evolution.csv"
    pdf_file=f"{folder_name}/training_performance.pdf"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])
        epochs = range(1,num_epochs+1)
        for epoch, loss in zip(epochs, losses):
            writer.writerow([epoch, loss])

    plt.figure(figsize=(12, 5))
    # Plotting training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()

    plt.savefig(pdf_file, format='pdf')


def add_config(section_, key_, value_, ):
    if section_ not in list(config.sections()):
        config.add_section(section_)
    config[section_][key_] = str(value_)
    filename = config["param"]["config_filename"]
    with open(filename, "w") as conf:
        config.write(conf)


def shuffle_dataset(original_list):
    set_seed()
    shuffled_list = original_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list

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

def load_data(dataset_dir,batch_size=16,num_samples_per_class=0,use_class_weights=True):
        set_seed()
        train_folder,val_folder,test_folder = dataset_dir[0],dataset_dir[1],dataset_dir[2]
        
        # Create datasets
        train_dataset = datasets.ImageFolder(train_folder, transform=transform(type_data="train"))
        test_dataset = datasets.ImageFolder(test_folder, transform=transform(type_data="test"))
        val_dataset = datasets.ImageFolder(val_folder, transform=transform(type_data="test"))    
        print(f"Train dataset: {len(train_dataset)} images")
        print(f"Validation dataset: {len(val_dataset)} images") 
        print(f"Test dataset: {len(test_dataset)} images")
        num_classes = len( train_dataset.classes)
        targets = [train_dataset.targets[i] for i in range(len(train_dataset))]
        indices = list(range(len(train_dataset)))


        # Create data loaders

        sampler = BalancedSampler(
            indices=indices,
            num_samples_per_class=num_samples_per_class,
            class_to_idx=train_dataset.class_to_idx,
            targets=targets
        )

        if num_samples_per_class==0:
             # Calculate class frequencies in the training dataset
            class_counts = [len(np.where(np.array(train_dataset.targets) == i)[0]) for i in range(len(train_dataset.classes))]

                # Calculate weights for each class based on inverse frequency
            weights = 1. / np.array(class_counts)

                # Create a weight array for each sample in the dataset based on its class label
            sample_weights = np.array([weights[label] for label in train_dataset.targets])
                
            if use_class_weights==True:
               
                # Create the WeightedRandomSampler
                train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

                # Create DataLoader for training and testing
                train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            else:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            train_loader = image_DataLoader(train_dataset,  batch_size=batch_size,sampler=sampler)
       
        val_loader = image_DataLoader(val_dataset,  batch_size=batch_size,shuffle=False) 
        test_loader = image_DataLoader(test_dataset,  batch_size=batch_size,shuffle=False)
        
        return num_classes, train_loader, val_loader, test_loader, train_dataset.classes,sample_weights
        


def split_image_dataset(data_path):
   
        # path to destination folders
    train_folder = os.path.join(data_path, 'train')
    val_folder = os.path.join(data_path, 'eval')
    test_folder = os.path.join(data_path, 'test')

    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png']

    # Create a list of image filenames in 'data_path'
    imgs_list = [filename for filename in os.listdir(data_path) if os.path.splitext(filename)[-1] in image_extensions]

    # Sets the random seed 
    set_seed()   
    # Shuffle the list of image filenames
    random.shuffle(imgs_list)

    # determine the number of images for each set
    train_size = int(len(imgs_list) * 0.7)
    val_size = int(len(imgs_list) * 0.15)
    test_size = int(len(imgs_list) * 0.15)

    # Create destination folders if they don't exist
    for folder_path in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Copy image files to destination folders
    for i, f in enumerate(imgs_list):
        if i < train_size:
            dest_folder = train_folder
        elif i < train_size + val_size:
            dest_folder = val_folder
        else:
            dest_folder = test_folder
        shutil.copy(os.path.join(data_path, f), os.path.join(dest_folder, f))
        print(f"Copying {f} to {dest_folder}")


    return train_folder,val_folder,test_folder 



def transform(type_data="train"):
    # Common preprocessing steps
    preprocessing = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if type_data == "train":
        # Additional augmentations for training
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        ]
        return transforms.Compose(augmentations + preprocessing)

    elif type_data == "test":
        # No augmentations for test data, only preprocessing
        return transforms.Compose(preprocessing)

    else:
        raise ValueError(f"Unsupported type_data: {type_data}. Use 'train' or 'test'.")



def save_training_details(train_loss, train_acc, test_acc, txt_file, pdf_file,model_name):
    """
    Save the provided metric values into a .txt file and generate plots saved into a PDF file.

    Args:
        train_loss (list): List of training loss values.
        train_acc (list): List of training accuracy values.
        test_acc (list): List of test accuracy values.
        txt_file (str): Path to the output .txt file for saving metrics.
        pdf_file (str): Path to the output PDF file for saving plots.
    """

    # Save the metrics into a .txt file
    with open(txt_file, "w") as f:
        f.write("Training Loss:\n")
        f.write(", ".join(map(str, train_loss)) + "\n\n")
        f.write("Training Accuracy:\n")
        f.write(", ".join(map(str, train_acc)) + "\n\n")
        f.write("Test Accuracy during training:\n")
        f.write(", ".join(map(str, test_acc)) + "\n\n")

    plt.figure(figsize=(10, 6))

    # Plot Train Loss
    plt.plot(range(len(train_loss)),train_loss, label='Train Loss', color='red', linestyle='--')

    # Plot Train Accuracy
    plt.plot(range(len(train_acc)), train_acc, label='Train Accuracy', color='blue')

    # Plot Test Accuracy
    plt.plot(range(len(test_acc)), test_acc, label='Test Accuracy', color='green')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title('Trainning stats for model:' + model_name )

    # Show legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Save the plot as a PDF file
    plt.savefig(pdf_file, format='pdf')

    # Show the plot
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, file_name="confusion_matrix.pdf"):
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix (optional, you can comment this line if you want raw counts)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Create a heatmap using seaborn for better visualization
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                     yticklabels=class_names, cbar=True)

    # Add labels and title
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)

    # Save the confusion matrix plot to a PDF file
    plt.savefig(file_name)
    plt.close()  # Close the plot after saving


def save_multiple_time_to_excel_with_date(cr, args):
    output_file_path = os.path.join(args.result_dir, f"result_for_{args.model_name}.xlsx")
    current_date = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(output_file_path):
        workbook = load_workbook(output_file_path)
        with pd.ExcelWriter(output_file_path, engine="openpyxl", mode="a") as writer:
            writer.book = workbook
            cr.to_excel(writer, sheet_name=current_date)
    else:
        cr.to_excel(output_file_path)

def graphdata_loader(graph_list,batch_size,type_data="train",ddp=True):
    set_seed()
  
    sampler=ImbalancedSampler(graph_list)
    if type_data == "train":
        dataset_loader = DataLoader(graph_list, batch_size=batch_size, sampler=sampler)
    else:
        dataset_loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())


    return dataset_loader

def Load_graphdata(dataset_source_path,type_graph="multi_graph"):
    set_seed()
    graph_list = []
    label_dict = {}
    # x_size=defaultdict(list)
    
    assert os.path.isdir(dataset_source_path), "The provided dataset_source_path is not a valid directory."

    # Get all file paths in the directory
    file_paths = [os.path.join(dataset_source_path, file_name) for file_name in os.listdir(dataset_source_path)]

    # Use ThreadPoolExecutor to load graphs in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Load all graphs concurrently
        results = executor.map(load_single_graph, file_paths)

    for idx,data in tqdm(enumerate(results)):
    
        if int(data.y.item()) not in label_dict.keys():
            label_dict[int(data.y.item())] = data.label_name
        # if data.x.shape[1] in x_size.keys():
        #     x_size[data.x.shape[1]].append(1)
        
        graph_list.append(data)
    # Sorting labels and printing dataset details
    labels = list(dict(sorted(label_dict.items())).values())   
    
    print(f"Graph dataset sample: {graph_list[0]}")
    
    feat_size_list = []	
    if type_graph=="multi_graphs":
        feat_size_list.append(data.x1.shape[1])
        feat_size_list.append(data.x2.shape[1])
        feat_size_list.append(data.x3.shape[1])
        feat_size_list.append(data.x4.shape[1])
        feat_size_list.append(data.x5.shape[1])
    else:
        feat_size_list.append(data.x.shape[1])
    
    return graph_list, feat_size_list, labels



def load_single_graph(file_path):
    """ Helper function to load a single graph from a file. """
    data = torch.load(file_path)
    return data



def calculate_running_time(start_time, end_time):
    # Convert string inputs to datetime objects
    fmt = "%H:%M"  # Format: HH:MM (24-hour format)
    start = datetime.strptime(start_time, fmt)
    end = datetime.strptime(end_time, fmt)
    
    # Calculate the time difference
    duration = end - start
    
    # Convert to hours
    hours = duration.total_seconds() / 3600  # Convert seconds to hours
    
    return hours




# Function to load image
def load_image(image_path):
    """ Load image and convert to tensor """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

def fix_feature_length(features, fixed_size=fixe_size, pad_value=0):
    
    """ Truncates or pads the feature vector to a fixed size """
    if len(features) > fixed_size:
        return features[:fixed_size]  # Truncate
    elif len(features) < fixed_size:
        return np.pad(features, (0, fixed_size - len(features)), constant_values=pad_value)  # Pad
    return features



def extract_superpixel_features(img_rgb, n_segments=100):
    """ Extract mean color features from superpixel segmentation """
    segments = slic(img_rgb, n_segments=n_segments, compactness=10)
    mean_colors = []

    for segment_label in np.unique(segments):
        mask = segments == segment_label
        mean_color = np.mean(img_rgb[mask], axis=0)
        mean_colors.extend(mean_color)
    features=np.array(mean_colors)
      #print(f"Superpixel feature size is: {features.shape}")
    return features

def extract_keypoint_features(gray_img, max_features=100):
    """ Extract ORB keypoints and enforce fixed size """
    orb = ORB(n_keypoints=max_features)
    orb.detect_and_extract(gray_img)
    keypoints = orb.descriptors
    
    if keypoints is None or len(keypoints) == 0:
        return np.zeros(max_features * 8)  # Default zero vector

    keypoints = keypoints[:max_features]
    features=keypoints.flatten()
     # print(f"Keypoint feature size is: {features.shape}")
    return features

def extract_region_adjacency_features(gray_img):
    """ Extract adjacency features using a region adjacency graph and fix length """
    labels = slic(gray_img, n_segments=50, compactness=10)
    edges = graph.rag_mean_color(gray_img, labels)

    edge_weights = [data['weight'] for _, _, data in edges.edges(data=True)]
    features=np.array(edge_weights)
    
      #print(f"Regions adjacency feature size is: {features.shape}")
    return features


def extract_grid_features(gray_img, grid_size=4):
    """ Divide the image into a grid and extract mean and std intensity per region """
    h, w = gray_img.shape
    grid_h, grid_w = h // grid_size, w // grid_size
    features = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            grid_patch = gray_img[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
            features.append(np.mean(grid_patch))
            features.append(np.std(grid_patch))
            
    features = np.array(features)   
     # print(f"grid feature size is: {features.size}")
    return features



def extract_feature_map(img_tensor):
    """ Extract CNN feature map using ResNet50 (pretrained) """
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer to keep feature maps

    with torch.no_grad():
        feature_map = resnet(img_tensor.unsqueeze(0))  # Pass through CNN
    features=np.array(feature_map.flatten().cpu().numpy())
    
      #print(f"Feature map feature size is: {features.shape}")    
    return features

def extract_mesh3d_features(gray_img, max_points=500):
    """ Convert image to a pseudo 3D mesh representation using edge detection """
    edges = sobel(gray_img)  # Compute edges (acts like height variations)
    mesh_points = np.column_stack(np.where(edges > 0))  # Get edge points
    features=np.array(mesh_points.flatten()[:max_points])
      #print(f"Mesh3d feature size is: {features.shape}")
    return features



class ImageFeatureDataset(Dataset):
    def __init__(self, image_paths, labels,feature_list, transform=None, fixed_size=0):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.feature_list=feature_list
        self.fixed_size=fixed_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        features_list=[]

        # Load image
        img, img_rgb = load_image(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract fixed-length features
        if "grid_features" in self.feature_list:
         grid_features = extract_grid_features(gray_img)
         if self.fixed_size!=0:
             grid_features=fix_feature_length(grid_features,self.fixed_size)
         features_list.append(grid_features)
         
        if "superpixel_features" in self.feature_list:
            superpixel_features = extract_superpixel_features(img_rgb)
            if self.fixed_size!=0:
              superpixel_features=fix_feature_length(superpixel_features,self.fixed_size)
            features_list.append(superpixel_features)
            
        if "keypoints_features" in self.feature_list:
             keypoint_features = extract_keypoint_features(gray_img)
             if self.fixed_size!=0:
                  keypoint_features=fix_feature_length(keypoint_features,self.fixed_size)
             features_list.append(keypoint_features)
        
        if "region_adjacency_features" in self.feature_list:
            region_adj_features = extract_region_adjacency_features(gray_img)
            if self.fixed_size!=0:
             region_adj_features=fix_feature_length(region_adj_features,self.fixed_size)
            features_list.append(region_adj_features)
            
        if "feature_map_features" in self.feature_list:
            feature_map = extract_feature_map(transforms.ToTensor()(img_rgb))
            if self.fixed_size!=0:
             feature_map=fix_feature_length(feature_map,self.fixed_size)
            features_list.append(feature_map)
            
        if "meash3d_features" in self.feature_list:
            mesh3d_features = extract_mesh3d_features(gray_img)
            if self.fixed_size!=0:
             mesh3d_features=fix_feature_length(mesh3d_features,self.fixed_size)
            features_list.append(mesh3d_features)

       
        # Convert image to tensor
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor()])
            img_tensor = transform(img_rgb)

        additional_features = np.hstack(tuple(features_list))

        additional_features_tensor = torch.tensor(additional_features, dtype=torch.float)
        return img_tensor, additional_features_tensor, torch.tensor(label, dtype=torch.long)




# Function to create dataset from a structured folder
def create_dataloader(data_dir, feature_list,fixed_size=0, batch_size=16,sufle=False):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # Get folder names (classes)
    
    # Encode class names to numeric labels
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):  # Ensure it's a directory
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):  # Ensure it's an image file
                    image_paths.append(os.path.join(class_path, img_name))
                    labels.append(class_name)

    # Convert labels to numeric
    labels = label_encoder.transform(labels)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = ImageFeatureDataset(image_paths = image_paths,
                                  labels = labels,
                                  feature_list = feature_list,
                                  fixed_size = fixed_size,
                                  transform=transform)
    
    dataloader = image_DataLoader(dataset, batch_size=batch_size, shuffle=sufle)

    return dataloader, label_encoder,class_names




class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for dealing with class imbalance.
        :param alpha: Weight for each class (tensor of shape [num_classes])
        :param gamma: Focusing parameter (higher = more focus on hard examples)
        :param reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: Model outputs before softmax (batch_size, num_classes)
        targets: Ground truth labels (batch_size)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # Probabilities for correct class

        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_factor = self.alpha[targets]  # Select weight for each target class
            focal_loss = alpha_factor * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def compute_class_weights(dataloader, num_classes, device="cpu"):
    """
    Computes class weights based on label frequencies from a DataLoader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing dataset batches.
        num_classes (int): Total number of classes in the dataset.
        device (str): Device where class weights should be stored (e.g., "cuda" or "cpu").

    Returns:
        torch.Tensor: Class weights tensor for Weighted Cross-Entropy Loss.
    """
    class_counts = Counter()

    # Iterate over the DataLoader to collect label frequencies
    for _, _, labels in dataloader:  # Assuming (images, features, labels) are returned
        class_counts.update(labels.tolist())  # Convert tensor to list and update counter

    # Convert counts to a tensor
    num_samples_per_class = torch.tensor(
        [class_counts[i] for i in range(num_classes)], dtype=torch.float32
    )

    # Compute weights: Inverse frequency (higher weight for underrepresented classes)
    class_weights = 1.0 / num_samples_per_class
    class_weights = class_weights / class_weights.sum()  # Normalize weights

    return class_weights.to(device)  # Move weights to the specified device
