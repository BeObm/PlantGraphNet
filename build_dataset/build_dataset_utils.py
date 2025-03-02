import torch
from torch_geometric.data import Data
from tqdm import tqdm
import os
from PIL import Image
from utils import *
import torch
import torchvision.models as models
import multiprocessing
from PIL import Image
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from skimage.segmentation import slic
# from skimage.future.graph import rag_mean_color
from scipy.spatial import Delaunay, Voronoi
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.utils import grid
from math import atan2, sqrt





def load_image(image_path):
    """ Load image as BGR and grayscale """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def save_graph(graph, output_path, label):
    """ Save PyTorch Geometric Data object with label """
    graph.y = torch.tensor([label], dtype=torch.long)  # Add label
    torch.save(graph, output_path)






def superpixel_graph(image_path,label):
    """
    Constructs a superpixel-based graph using SLIC segmentation and RAG.
    """
    img, _ = load_image(image_path)
    labels = slic(img, n_segments=50, compactness=10, start_label=0)
    num_segments = labels.max() + 1

    # Compute node features (average color per segment)
    avg_colors = np.array([img[labels == i].mean(axis=0) for i in range(num_segments)])
    x = torch.tensor(avg_colors, dtype=torch.float)

    # Compute region adjacency graph
    rag = rag_mean_color(img, labels)
    edges = [[n1, n2] for n1, n2 in rag.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).T

    # Compute edge attributes (color difference)
    edge_attr = torch.tensor(
        [rag[n1][n2]['weight'] for n1, n2 in rag.edges()],
        dtype=torch.float
    ).unsqueeze(1)  # Ensure shape [num_edges, 1]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))



def keypoint_graph(image_path,label):
    """
    Constructs a keypoint-based graph using SIFT and nearest neighbors.
    """
    img, gray = load_image(image_path)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    if des is None:
        return Data(x=torch.zeros((0, 128)), edge_index=torch.empty((2, 0), dtype=torch.long))

    # Convert keypoints to a PyTorch tensor
    points = np.array([k.pt for k in kp])
    x = torch.tensor(des, dtype=torch.float)

    # Nearest neighbors (k=5)
    nbrs = NearestNeighbors(n_neighbors=5).fit(points)
    adjacency_matrix = nbrs.kneighbors_graph(points, 5, mode='connectivity').tocoo()
    edge_index = torch.tensor(np.vstack([adjacency_matrix.row, adjacency_matrix.col]), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=torch.tensor([1]*edge_index.shape[1],dtype=torch.float), y=torch.tensor([label]))


def region_adjacency_graph(image_path,label):
    """
    Constructs a region adjacency graph using superpixels and RAG.
    """
    img, _ = load_image(image_path)
    labels = slic(img, n_segments=20, compactness=20, start_label=0)
    rag = rag_mean_color(img, labels)

    num_regions = labels.max() + 1
    x = torch.tensor([img[labels == i].mean(axis=0) for i in range(num_regions)], dtype=torch.float)

    edges = [[n1, n2] for n1, n2 in rag.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).T

    # Edge weights based on color similarity
    edge_attr = torch.tensor(
        [rag[n1][n2]['weight'] for n1, n2 in rag.edges()],
        dtype=torch.float
    ).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))


def feature_map_graph(image_path,label):
    """
    Converts an image into a feature map-based graph.
    """
    img, _ = load_image(image_path)
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
    feat = conv(img_tensor)[0]  # [C, H, W]

    C, H, W = feat.shape
    x = feat.view(C, -1).T  # Flatten feature map into nodes
    edge_index, _ = grid(H, W)

    return Data(x=x, edge_index=edge_index, edge_attr=torch.tensor([1]*edge_index.shape[1],dtype=torch.float), y=torch.tensor([label]))



def mesh3d_graph(image_path,label):
    """
    Constructs a 3D mesh graph from a depth map.
    """
    img, _ = load_image(image_path)
    depth = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    H, W = depth.shape

    coords_2d = np.indices((H, W)).reshape(2, -1).T
    x = torch.tensor(np.hstack([coords_2d, depth.reshape(-1, 1)]), dtype=torch.float)

    edge_index, _ = grid(H, W)

    return Data(x=x, edge_index=edge_index, edge_attr=torch.tensor([1]*edge_index.shape[1],dtype=torch.float), y=torch.tensor([label]))



def grid_graph(image_path,label, connectivity=4):
    """
    Constructs a grid-based graph with 4-neighborhood or 8-neighborhood connectivity.
    """
    img, _ = load_image(image_path)
    H, W, C = img.shape
    x = torch.tensor(img.reshape(-1, C), dtype=torch.float)

    edges = []
    for r in range(H):
        for c in range(W):
            node_idx = r * W + c

            # 4-neighborhood
            if r > 0:   edges.append((node_idx, (r-1) * W + c))
            if r < H-1: edges.append((node_idx, (r+1) * W + c))
            if c > 0:   edges.append((node_idx, r * W + (c-1)))
            if c < W-1: edges.append((node_idx, r * W + (c+1)))

            # 8-neighborhood
            if connectivity == 8:
                if r > 0 and c > 0:   edges.append((node_idx, (r-1) * W + (c-1)))
                if r > 0 and c < W-1: edges.append((node_idx, (r-1) * W + (c+1)))
                if r < H-1 and c > 0: edges.append((node_idx, (r+1) * W + (c-1)))
                if r < H-1 and c < W-1: edges.append((node_idx, (r+1) * W + (c+1)))

    edge_index = torch.tensor(edges, dtype=torch.long).T

    return Data(x=x, edge_index=edge_index,edge_attr=torch.tensor([1]*edge_index.shape[1],dtype=torch.float), y=torch.tensor([label]))






def delaunay_graph(image_path, label):
    """
    Constructs a graph using Delaunay triangulation on SIFT keypoints.
    Args:
        image_path (str): Path to the input image.
        label (int): Label for the graph.
    Returns:
        Data: PyTorch Geometric graph with x, edge_index, edge_attr, and y.
    """
    # Load and process image
    _, gray = load_image(image_path)
    
    # Detect SIFT keypoints
    sift = cv2.SIFT_create()
    kp, _ = sift.detectAndCompute(gray, None)
    if len(kp) == 0:
        raise ValueError("No keypoints detected in the image!")

    # Extract keypoint coordinates
    points = np.array([k.pt for k in kp], dtype=np.float32)
    
    # Convert keypoints to a PyTorch tensor (node feature matrix)
    X = torch.tensor(points, dtype=torch.float)
    
    # Perform Delaunay triangulation
    tri = Delaunay(points)
    
    # Extract unique undirected edges
    edges = set()
    for simplex in tri.simplices:  # each simplex is a triangle (3 vertices)
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            edge = tuple(sorted((simplex[i], simplex[j])))
            edges.add(edge)
    
    # Convert edge list to edge_index tensor [2, num_edges]
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    
    # Compute edge attributes (distance & angle)
    edge_attr_list = []
    for i, j in edges:
        xi, yi = points[i]
        xj, yj = points[j]
        dist = sqrt((xj - xi)**2 + (yj - yi)**2)
        angle = atan2(yj - yi, xj - xi)
        edge_attr_list.append([dist, angle])
    
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    # Create PyG Data object
    data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))
    return data





def voronoi_graph(image_path, label):
    """
    Constructs a graph using a Voronoi diagram on SIFT keypoints.
    Args:
        image_path (str): Path to the input image.
        label (int): Label for the graph.
    Returns:
        Data: PyTorch Geometric graph with x, edge_index, edge_attr, and y.
    """
    # Load and process image
    _, gray = load_image(image_path)
    
    # Detect SIFT keypoints
    sift = cv2.SIFT_create()
    kp, _ = sift.detectAndCompute(gray, None)
    if len(kp) == 0:
        raise ValueError("No keypoints detected in the image!")

    # Extract keypoint coordinates
    points = np.array([k.pt for k in kp], dtype=np.float32)
    
    # Convert keypoints to a PyTorch tensor (node feature matrix)
    X = torch.tensor(points, dtype=torch.float)
    
    # Compute Voronoi diagram
    vor = Voronoi(points)
    
    # Extract Voronoi edges (ridge_points gives pairs of point indices)
    edges = set()
    for (p, q) in vor.ridge_points:
        edge = tuple(sorted((p, q)))
        edges.add(edge)
    
    # Convert edge list to edge_index tensor [2, num_edges]
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    
    # Compute edge attributes (distance & angle)
    edge_attr_list = []
    for i, j in edges:
        xi, yi = points[i]
        xj, yj = points[j]
        dist = sqrt((xj - xi)**2 + (yj - yi)**2)
        angle = atan2(yj - yi, xj - xi)
        edge_attr_list.append([dist, angle])
    
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    # Create PyG Data object
    data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))
    return data