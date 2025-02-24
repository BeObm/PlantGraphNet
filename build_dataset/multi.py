import cv2
import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx

# Load image
img = cv2.imread(r'C:\Users\au783153\OBM\CODES\HeathlandSpeciesClassifier\dataset\images\lidl\amm\im51_44_9.1.0.jpg', cv2.IMREAD_GRAYSCALE)

# --- Function to extract local pixel features (patch around keypoint) ---
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
    return torch.tensor(patch).view(1, patch_size, patch_size)  # Reshape for PyTorch (1, patch_size, patch_size)

# --- Function to construct the graph with pixel features ---
def construct_graph_with_pixel_features(keypoints, descriptors, img, distance_threshold=50,similarity_treshold=0.7, patch_size=32, type="sift"):
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
                        descriptor_distance = cv2.norm(descriptors[i], descriptors[j], cv2.NORM_HAMMING)    
                    else:
                        #  descriptor_distance = np.linalg.norm(descriptors[i] - descriptors[j])
                        descriptor_distance = cv2.norm(descriptors[i], descriptors[j], cv2.NORM_L2) 
                    if descriptor_distance < similarity_treshold:  # Threshold for similarity 
                        print(f"Adding edge between {i} and {j} with distance {dist} and descriptor distance {descriptor_distance}")
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
    return data


# --- SIFT Example (Direct Access) ---
def extract_sift(img):
    sift = cv2.SIFT_create()  # OpenCV 4.4+ no need for contrib module
    keypoints, descriptors = sift.detectAndCompute(img, None)
    descriptors = cv2.normalize(descriptors, None, norm_type=cv2.NORM_L2)
    return keypoints, descriptors

# --- ORB Example ---
def extract_orb(img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

# --- FAST Example ---
def extract_fast(img):
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(img, None)
    descriptors = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return keypoints, descriptors

# --- AKAZE Example ---
def extract_akaze(img):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(img, None)
    return keypoints, descriptors





import numpy as np




# --- Harris Corner Detection Example ---
# --- Harris Corner Detection Example ---
# --- Harris Corner Detection Example ---
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
    
    # Create keypoints from corner locations (using size=1 for Harris corners)
    keypoints = [cv2.KeyPoint(float(corner[1]), float(corner[0]), 1) for corner in corners]  # Set size to 1
    
    # Create descriptors (simple example: just the surrounding pixels as descriptors)
    descriptors = np.array([extract_patch_features(img, kp).numpy().flatten() for kp in keypoints])
    
    return keypoints, descriptors

print("\nHarris Graph Data:")

# --- Harris Example (Add it into the graph construction process) ---
keypoints_harris, descriptors_harris = extract_harris(img)
descriptors = (descriptors_harris - descriptors_harris.min()) / (descriptors_harris.max() - descriptors_harris.min())
data_harris = construct_graph_with_pixel_features(keypoints_harris, descriptors, img)

# --- Construct Graph for each method with pixel features ---
print("\nHarris Graph Data:")
print("Nodes:", data_harris.num_nodes)
print("Edges:", data_harris.num_edges)
print("Node Features Shape:", data_harris.x.shape)
print(data_harris)