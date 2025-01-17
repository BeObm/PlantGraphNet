import torch
from torch_geometric.data import Data
from tqdm import tqdm
import os
from utils import *
import torch
import torchvision.transforms as transform
import torchvision.models as models
from PIL import Image
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt




# Build the PyTorch Geometric dataset using grid-based approach
def build_dataset(dataset_path, output_path, nb_per_class=200):
    dataset = []
    class_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for label, class_folder in enumerate(class_folders):
        pbar = tqdm(len(class_folders))
        pbar.set_description(f"Contructing graph data for Class #{label}: {class_folder} ... ")
        class_path = os.path.join(dataset_path, class_folder)
        if nb_per_class==0:
          image_files = shuffle_dataset([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))])
        else:
          image_files = shuffle_dataset([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))])[:nb_per_class]
        a = 1
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            data = image_to_graph(img_path, label)
            dataset.append(data)
            if a < 3:
                plot_image_with_nodes(img_path, data, f"{config['param']['result_folder']}/ImageAndGraph/{label}/{a}")
                a += 1
        pbar.set_description(f"Contructed {len(image_files)} graphs  for Class #{label}: {class_folder} ")
        pbar.update(1)
    torch.save(dataset, output_path)


def image_to_graph(img_path, label):
    img = io.imread(img_path)
    img_tensor = transforms.ToTensor()(img)
    x, edge_index = get_node_features_and_edge_list(img_tensor)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

def get_node_features_and_edge_list(image):
    """
    Convert an image into a graph representation using an edge list.
    Parameters:
        image (torch.Tensor): An input image of shape (height, width, channels),
                              where 'channels' is typically 3 for RGB images.
    Returns:
        edges (list of tuple): List of edges, where each edge is a tuple (node1, node2).
        node_features (torch.Tensor): Node features of the graph of size (num_pixels, channels).
    """
    image = validate_image(image)
    height, width, channels = image.shape

    # Flatten the image into a list of nodes
    node_features = image.view(-1, channels)  # Shape: (num_pixels, channels)

    # Create an edge list for the graph
    edges = compute_edges(height, width)

    return  node_features, edges


def validate_image(image):
    """Validate the input image."""
    if isinstance(image, np.ndarray):
        # Convert numpy array to torch tensor if needed
        image = torch.from_numpy(image)
    if image.dim() != 3:
        raise ValueError("Input image must have 3 dimensions: (height, width, channels)")
    return image


def compute_edges(height, width):
    """Generate an edge list for a graph based on 8-connectivity."""
    edges = []
    for i in range(height):
        for j in range(width):
            current_index = pixel_to_index(i, j, width)
            neighbors = [
                (i - 1, j),  # Top
                (i + 1, j),  # Bottom
                (i, j - 1),  # Left
                (i, j + 1),  # Right
                (i - 1, j - 1),  # Top-left
                (i - 1, j + 1),  # Top-right
                (i + 1, j - 1),  # Bottom-left
                (i + 1, j + 1),  # Bottom-right
            ]
            # Collect valid edges
            for ni, nj in neighbors:
                if 0 <= ni < height and 0 <= nj < width:  # Ensure within bounds
                    neighbor_index = pixel_to_index(ni, nj, width)
                    edges.append((current_index, neighbor_index))
    return edges



def pixel_to_index(x, y, width):
    """Convert 2D pixel coordinates to a flattened index."""
    return x * width + y


