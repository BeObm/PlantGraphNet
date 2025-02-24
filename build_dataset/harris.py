from skimage import color
from skimage.feature import corner_harris, corner_peaks
from torch_geometric.data import Data
from torchvision import transforms
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import multiprocessing
from utils import *
import os
from tqdm import tqdm
import cv2


# Build the PyTorch Geometric dataset using Harris conner detection approach
def build_dataset(dataset_path, args,type_dataset,apply_transform=True):
    
    nb_per_class=args.images_per_class
    connectivity = args.connectivity
    use_image_feats = args.use_image_feats
  
    dataset = []
    class_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    graph_dataset_dir = f"{config['param']['graph_dataset_folder']}/{type_dataset}"
    os.makedirs(graph_dataset_dir, exist_ok=True)
    set_seed()

    for label, class_folder in tqdm(enumerate(class_folders)):
        print(f"Contructing graph data for Class #{label}: {class_folder} ... \n ")

        class_path = os.path.join(dataset_path, class_folder)
        if nb_per_class==0:
          image_files = shuffle_dataset([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))])
        else:
          image_files = shuffle_dataset([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))])[:nb_per_class]
        a = 1
        
        with multiprocessing.Pool() as pool:
            pool.starmap(image_to_graph, [(os.path.join(class_path, img_file), label, class_folder, connectivity, apply_transform, f"{graph_dataset_dir}/{type_dataset}_{label}_{idx}.pt",use_image_feats) for idx,img_file in enumerate(image_files)])
       
        # for idx,img_file in enumerate(image_files):
        #     img_path = os.path.join(class_path, img_file)
        #     image_to_graph(image_path=img_path,
        #                    label=label,
        #                    label_name=class_folder,
        #                    apply_transforms=apply_transform,
        #                    output_path=f"{graph_dataset_dir}/{label}_{idx}.pt")

        print(f"Contructed {len(image_files)} graphs  for Class #{label}: {class_folder} \n",flush=True)





def image_to_graph(image_path, label, label_name, connectivity, apply_transforms, output_path, use_image_feats, k=0.04, threshold=0.01):
    """
    Convert an image to a graph representation using Harris corner detection and optional image features.

    Args:
        image_path (str): Path to the input image.
        label (int): Label associated with the image.
        label_name (str): Name of the label.
        connectivity (str): Type of connectivity ('4-connectivity' or '8-connectivity').
        apply_transforms (bool): Whether to apply data augmentation or transformation to the image.
        output_path (str): Path where the graph data should be saved.
        use_image_feats (bool): Whether to include image features (e.g., from a pre-trained model).
        k (float, optional): Harris corner detection constant (default 0.04).
        threshold (float, optional): Threshold for corner detection (default 0.01).

    Returns:
        Data: A PyTorch Geometric Data object representing the graph.
    """
    
    # Step 1: Load image in grayscale and resize
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")
    image = cv2.resize(image, (128, 128))
    height, width = image.shape

    # Step 2: Apply Harris corner detection
    harris_response = cv2.cornerHarris(image, blockSize=3, ksize=3, k=k)

    # Step 3: Threshold to extract strong corners
    corners = np.zeros_like(harris_response, dtype=np.uint8)
    corners[harris_response > threshold * harris_response.max()] = 1

    # Step 4: Extract (x, y) positions of corner points
    corner_positions = np.argwhere(corners == 1)
    corner_indices = {tuple(pos): idx for idx, pos in enumerate(corner_positions)}
    
    # Step 5: Create edges based on neighbor connectivity
    edges = create_edges(corner_positions, corner_indices, connectivity, height, width, corners)

    # Step 6: Convert corner positions to node features (pixel intensity at each corner)
    node_features = extract_node_features(corner_positions, image)

    # Step 7: Optionally, apply image transformations (e.g., for data augmentation)
    img = apply_image_transforms(image_path, apply_transforms)

    # Step 8: Create PyTorch Geometric Data object
    data = create_data_object(node_features, edges, label, img, use_image_feats, label_name)

    # Step 9: Save the graph data
    torch.save(data, output_path)

    return data


def create_edges(corner_positions, corner_indices, connectivity, height, width, corners):
    """
    Create edges based on 4-connectivity or 8-connectivity.

    Args:
        corner_positions (ndarray): Coordinates of corner points.
        corner_indices (dict): Mapping from corner coordinates to indices.
        connectivity (str): Type of connectivity ('4-connectivity' or '8-connectivity').
        height (int): Height of the image.
        width (int): Width of the image.
        corners (ndarray): Harris corner detection results.

    Returns:
        list: List of edges as tuples (from_node_idx, to_node_idx).
    """
    edges = []
    for pos in corner_positions:
        i, j = pos

        # Define neighbor offsets based on chosen connectivity
        if connectivity == '8-connectivity':
            neighbors = [
                (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
                (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)
            ]
        elif connectivity == '4-connectivity':
            neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        else:
            raise ValueError(f"Invalid connectivity '{connectivity}'. Choose '4-connectivity' or '8-connectivity'.")

        # Check each neighbor and add an edge if it's a valid corner
        for ni, nj in neighbors:
            if 0 <= ni < height and 0 <= nj < width and corners[ni, nj] == 1:
                edges.append((corner_indices[tuple(pos)], corner_indices[(ni, nj)]))

    return edges


def extract_node_features(corner_positions, image):
    """
    Extract pixel intensity at corner positions as node features.

    Args:
        corner_positions (ndarray): Coordinates of corner points.
        image (ndarray): Grayscale image.

    Returns:
        torch.Tensor: Node features (pixel intensities) as a tensor.
    """
    node_features = [image[tuple(pos)] for pos in corner_positions]
    return torch.tensor(node_features, dtype=torch.float32)


def apply_image_transforms(image_path, apply_transforms):
    """
    Apply image transformations (e.g., data augmentation).

    Args:
        image_path (str): Path to the image.
        apply_transforms (bool): Whether to apply transformations.

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    image = Image.open(image_path).convert('RGB')

    if apply_transforms:
        transform_pipeline = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_pipeline = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    img = transform_pipeline(image).unsqueeze(dim=0)  # Add batch dimension
    return img


def create_data_object(node_features, edges, label, img, use_image_feats, label_name):
    """
    Create a PyTorch Geometric Data object.

    Args:
        node_features (torch.Tensor): Node features.
        edges (list): List of edges.
        label (int): Label associated with the image.
        img (torch.Tensor): Transformed image tensor (optional).
        use_image_feats (bool): Whether to include image features.
        label_name (str): Name of the label.

    Returns:
        Data: PyTorch Geometric Data object.
    """
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    if use_image_feats:
        data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), image_features=img, label_name=label_name)
    else:
        data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), label_name=label_name)

    return data
