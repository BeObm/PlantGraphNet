from collections import defaultdict
import torch
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
from torchvision import datasets, transforms
from torch_geometric.loader import DataLoader, ImbalancedSampler
import concurrent.futures
from torch.utils.data import  SubsetRandomSampler,WeightedRandomSampler
from torch.utils.data import DataLoader as image_DataLoader
import random
import os
from datetime import datetime

import shutil

config = ConfigParser()
RunCode = dates = datetime.now().strftime("%d-%m_%Hh%M")
project_root_dir = os.path.abspath(os.getcwd())

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
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR
            )
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

def Load_graphdata(dataset_source_path):
    set_seed()
    graph_list = []
    label_dict = {}
    x_size=[]
    
    assert os.path.isdir(dataset_source_path), "The provided dataset_source_path is not a valid directory."

    # Get all file paths in the directory
    file_paths = [os.path.join(dataset_source_path, file_name) for file_name in os.listdir(dataset_source_path)]

    # Use ThreadPoolExecutor to load graphs in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Load all graphs concurrently
        results = executor.map(load_single_graph, file_paths)

    for data in results:
        print("This the graph data.x size ",data.x.shape[1])
    
        if int(data.y.item()) not in label_dict.keys():
            label_dict[int(data.y.item())] = data.label_name
        if int(data.x.shape[1]) not in x_size:
            x_size.append(data.x)
        
        graph_list.append(data)
    print(f"Distinct x size: {len(x_size)}")
    # Sorting labels and printing dataset details
    labels = list(dict(sorted(label_dict.items())).values())    
    feat_size = data.x.shape[1]

    return graph_list, feat_size, labels



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
