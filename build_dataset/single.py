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


# Build the PyTorch Geometric dataset using grid-based approach
def build_dataset(dataset_path, args,type_dataset,apply_transform=True):
    
    nb_per_class=args.images_per_class
    node_detector = args.type_node_detector
    use_image_feats = args.use_image_feats
    dataset = []
    class_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    graph_dataset_dir = f"{config['param']['graph_dataset_folder']}/{args.type_node_detector}/{type_dataset}"
    os.makedirs(graph_dataset_dir, exist_ok=True)

    for label, class_folder in tqdm(enumerate(class_folders)):
        print(f"Contructing graph data for Class #{label}: {class_folder} ... \n ")
        
        class_path = os.path.join(dataset_path, class_folder)
        if nb_per_class==0:
          image_files = shuffle_dataset([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))])
        else:
          image_files = shuffle_dataset([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))])[:nb_per_class]
        a = 1
        with multiprocessing.Pool() as pool:
            pool.starmap(image_to_graph, [(os.path.join(class_path, img_file), label, class_folder, node_detector, apply_transform, f"{graph_dataset_dir}/{label}_{idx}.pt",use_image_feats) for idx,img_file in enumerate(image_files)])
        
        # for idx,img_file in enumerate(image_files):
        #     img_path = os.path.join(class_path, img_file)
        #     image_to_graph(img_path=img_path,
        #                    label=label,
        #                    label_name=class_folder,
        #                    apply_transforms=apply_transform,
                        #    output_path=f"{graph_dataset_dir}/{label}_{idx}.pt")

        print(f"Contructed {len(image_files)} graphs  for Class #{label}: {class_folder} \n")




def image_to_graph(img_path, label,label_name,node_detector,apply_transforms=True, output_path="data/graph_data.pt", use_image_feats=False):
    print(f"Processing image: {img_path}")
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2=Image.open(img_path).convert('RGB')

    # if apply_transforms:
    #     transform_pipeline= transform(type_data="train")
    #     img = transform_pipeline(img)
    # else:
    #     transform_pipeline = transform(type_data="test")
    #     img = transform_pipeline(img)
        
    if node_detector=="harris":   
          keypoints, descriptors0 = extract_harris(img)
    elif node_detector=="sift":
        keypoints, descriptors0 = extract_sift(img)
    elif node_detector=="orb":
        keypoints, descriptors0 = extract_orb(img)
    elif node_detector=="fast":
        keypoints, descriptors0 = extract_fast(img)
    elif node_detector=="akaze":
        keypoints, descriptors0 = extract_akaze(img)
    else:
        raise ValueError(f"Unknown node detector: {node_detector}")
    
    try:
    
        descriptors = (descriptors0 - descriptors0.min()) / (descriptors0.max() - descriptors0.min())
    except:
        print(f"Error in processing image: {img_path}")
        print(f"Descriptors: {descriptors0}")
        print(f" Descriptor type: {type(descriptors0)}")
        raise ValueError(f"Error in processing image: {img_path}")

    x, edge_index, edge_attr= construct_graph_with_pixel_features(keypoints, descriptors, img,distance_threshold=50,similarity_treshold=100, patch_size=32, type=node_detector)
    y = torch.tensor([label], dtype=torch.long)
    if use_image_feats==True:
         data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, image_features=img2.unsqueeze(dim=0),label_name=label_name)
    else:
        data=Data(x=x, edge_index=edge_index, y=y,label_name=label_name)
        
    torch.save(data, output_path)




def construct_graph_with_pixel_features(keypoints, descriptors, img, distance_threshold=50,similarity_treshold=100, patch_size=32, type="sift"):
    """
    Constructs a graph with keypoints as nodes, incorporating pixel features from patches around the keypoints.
    
    Parameters:
        keypoints (list): List of detected keypoints.
        descriptors (numpy.ndarray): Descriptors corresponding to the keypoints.
        img (ndarray): The original image.
        distance_threshold (float): Threshold for spatial proximity to connect keypoints.
        patch_size (int): Size of the patch around each keypoint to extract as node feature.
        
    Returns:
        Data: PyTorch Geometric graph data with pixel features.
    """
    G = nx.Graph()

    # Add nodes (keypoints) to the graph with pixel features
    for i, kp in enumerate(keypoints):
        pixel_feature = extract_patch_features(img, kp, patch_size)
        position = torch.tensor([kp.pt[0], kp.pt[1]], dtype=torch.float32)  # Position feature
        
        # Combine pixel feature and position
        node_feature = torch.cat((pixel_feature.view(-1), position))  # Flatten the patch and concatenate position
        G.add_node(i, x=node_feature)

    # Create edges based on proximity and/or descriptor similarity
    for i, kp1 in enumerate(keypoints):
        for j, kp2 in enumerate(keypoints):
            if i < j:
                dist = np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt))
                if dist < distance_threshold:
                    # Calculate descriptor distance (Euclidean distance between descriptors)
                    if type=="orb":
                        descriptor_distance = cv2.norm(descriptors[i], descriptors[j], normType=cv2.NORM_L2)    
                    else:
                        #  descriptor_distance = np.linalg.norm(descriptors[i] - descriptors[j])
                        descriptor_distance = cv2.norm(descriptors[i], descriptors[j], cv2.NORM_L2) 
                    if descriptor_distance < similarity_treshold:  # Threshold for similarity 
    
                        G.add_edge(i, j, weight=dist)

    # Extract edge information for PyTorch Geometric
    edge_index = []
    edge_attr = []
    
    for i, j, data in G.edges(data=True):
        edge_index.append([i, j])
        edge_attr.append(data['weight'])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    # Extract node features (concatenated pixel features and position)
    x = torch.stack([G.nodes[i]['x'] for i in range(len(keypoints))], dim=0)

    # Create a PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(f"Data: {data}")
    return x, edge_index, edge_attr


def extract_sift(img):
    sift = cv2.SIFT_create()  # OpenCV 4.4+ no need for contrib module
    keypoints, descriptors = sift.detectAndCompute(img, None)
    descriptors = cv2.normalize(descriptors, None, norm_type=cv2.NORM_L2)
    return keypoints, descriptors

def extract_orb(img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def extract_fast(img):
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(img, None)
    descriptors = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return keypoints, descriptors

def extract_akaze(img):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(img, None)
    return keypoints, descriptors


def extract_harris(img, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """
    Detect corners using Harris Corner Detection.
    
    Parameters:
        img (ndarray): The grayscale image.
        block_size (int): The size of the neighborhood considered for corner detection.
        ksize (int): Aperture parameter for the Sobel operator.
        k (float): Harris detector free parameter.
        threshold (float): Threshold to consider a corner.
        
    Returns:
        keypoints (list): List of keypoints (corners).
        descriptors (ndarray): Descriptors (pixel values around keypoints).
    """
    # Harris Corner detection
    img_harris = cv2.cornerHarris(img, block_size, ksize, k)
    
    # Normalize and threshold Harris response to get the corners
    img_harris = cv2.dilate(img_harris, None)
    corners = np.argwhere(img_harris > threshold * img_harris.max())
    
    # Create keypoints from corner locations
    keypoints = [cv2.KeyPoint(corner[1], corner[0], _size=1) for corner in corners]
    
    # Create descriptors (simple example: just the surrounding pixels as descriptors)
    descriptors = np.array([extract_patch_features(img, kp).numpy().flatten() for kp in keypoints])
    
    return keypoints, descriptors

def extract_patch_features(img, keypoint, patch_size=32):
    """
    Extract a patch of pixels centered around the keypoint.
    
    Parameters:
        img (ndarray): Input image (grayscale or color).
        keypoint (cv2.KeyPoint): Keypoint from which to extract the patch.
        patch_size (int): Size of the patch to extract.
        
    Returns:
        patch (torch.Tensor): Normalized pixel values from the patch.
    """
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    
    # Define patch boundaries
    half_size = patch_size // 2
    x1, x2 = max(x - half_size, 0), min(x + half_size, img.shape[1])
    y1, y2 = max(y - half_size, 0), min(y + half_size, img.shape[0])
    
    # Extract the patch
    patch = img[y1:y2, x1:x2]
    
    # If the patch size is smaller than desired, pad it with zeros
    if patch.shape != (patch_size, patch_size):
        patch = cv2.copyMakeBorder(patch, 0, patch_size - patch.shape[0], 0, patch_size - patch.shape[1], cv2.BORDER_CONSTANT, value=0)

    # Normalize the patch (optional: normalization helps neural networks)
    patch = patch.astype(np.float32) / 255.0  # Normalize to [0, 1] range
    # return torch.tensor(patch).view(1, patch_size, patch_size)  # Reshape for PyTorch (1, patch_size, patch_size)
    return torch.tensor(patch)  



def pixel_to_index(x, y, width):
    """Convert 2D pixel coordinates to a flattened index."""
    return x * width + y

