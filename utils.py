
import torch
from configparser import ConfigParser
import os.path as osp
import os
import csv
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from skimage import io
import random
import numpy as np
from torchvision import transforms

config = ConfigParser()
RunCode = dates = datetime.now().strftime("%d-%m_%Hh%M")
project_root_dir = os.path.abspath(os.getcwd())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_config_file(dataset_name,type_graph):
    configs_folder = osp.join(project_root_dir, f'results/{dataset_name}/{RunCode}')
    os.makedirs(configs_folder, exist_ok=True)
    config_filename = f"{configs_folder}/ConfigFile_{RunCode}.ini"
    graph_filename = f"{project_root_dir}/dataset/graphs"
    os.makedirs(graph_filename, exist_ok=True)
    config["param"] = {
        'config_filename': config_filename,
        "dataset_name": dataset_name,
        'type_graph': type_graph,
        "image_dataset_root": f"{project_root_dir}/dataset/images/{dataset_name}",
        "graph_dataset_name": f"{graph_filename}/{type_graph}/{dataset_name}.pt",
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

def transforme():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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

def plot_and_save_training_performance(num_epochs, losses, accuracies):
    filename = f"{config['param']['result_folder']}/training_performance.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss', 'Accuracy'])
        epochs = range(1,num_epochs+1)
        for epoch, loss, accuracy in zip(epochs, losses, accuracies):
            writer.writerow([epoch, loss, accuracy])

    plt.figure(figsize=(12, 5))
    # Plotting training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    # Plotting training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')

    plt.tight_layout()
    plt.show()
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


