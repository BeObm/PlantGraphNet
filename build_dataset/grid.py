import torch
from torch_geometric.data import Data
from tqdm import tqdm
import os
from PIL import Image
from utils import *
import torch
import torchvision.models as models
from PIL import Image
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Build the PyTorch Geometric dataset using grid-based approach
def build_dataset(dataset_path, nb_per_class=200,apply_transform=True):
    dataset = []
    class_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    graph_dataset_dir = f"{config['param']['graph_dataset_folder']}"


    for label, class_folder in tqdm(enumerate(class_folders)):
        print(f"Contructing graph data for Class #{label}: {class_folder} ... \n ")

        class_path = os.path.join(dataset_path, class_folder)
        if nb_per_class==0:
          image_files = shuffle_dataset([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))])
        else:
          image_files = shuffle_dataset([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))])[:nb_per_class]
        a = 1
        for idx,img_file in enumerate(image_files):
            img_path = os.path.join(class_path, img_file)
            image_to_graph(img_path=img_path,
                           label=label,
                           label_name=class_folder,
                           apply_transforms=apply_transform,
                           output_path=f"{graph_dataset_dir}/{label}_{idx}.pt")

        print(f"Contructed {len(image_files)} graphs  for Class #{label}: {class_folder} \n")




def image_to_graph(img_path, label,label_name,apply_transforms=True, output_path="data/graph_data.pt"):
    img = Image.open(img_path).convert('RGB')

    if apply_transforms:
        transform_pipeline= transform(type_data="train")
        img = transform_pipeline(img).unsqueeze(dim=0)
    else:
        transform_pipeline = transform(type_data="test")
        img = transform_pipeline(img).unsqueeze(dim=0)

        # img = torch.from_numpy(np.transpose(img, (2, 0, 1))).to(dtype=torch.float)
    x, edge_index = get_node_features_and_edge_list(img)
    y = torch.tensor([label], dtype=torch.long)
    data=Data(x=x, edge_index=edge_index, y=y, image_features=img,label_name=label_name)
    print(data)
    torch.save(data, output_path)

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
    channels, height, width = image.shape

    # Flatten the image into a list of nodes
    node_features = image.view(-1, channels)  # Shape: (num_pixels, channels)

    # Create an edge list for the graph
    edges = compute_edges(height, width)

    return  node_features, edges


def validate_image(image):

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
    return torch.tensor(edges,dtype=torch.long)



def pixel_to_index(x, y, width):
    """Convert 2D pixel coordinates to a flattened index."""
    return x * width + y


