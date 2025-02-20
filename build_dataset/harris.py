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


    for label, class_folder in tqdm(enumerate(class_folders)):
        print(f"Contructing graph data for Class #{label}: {class_folder} ... \n ")

        class_path = os.path.join(dataset_path, class_folder)
        if nb_per_class==0:
          image_files = shuffle_dataset([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))])
        else:
          image_files = shuffle_dataset([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))])[:nb_per_class]
        a = 1
        
        with multiprocessing.Pool() as pool:
            pool.starmap(image_to_graph, [(os.path.join(class_path, img_file), label, class_folder, connectivity, apply_transform, f"{graph_dataset_dir}/{label}_{idx}.pt",use_image_feats) for idx,img_file in enumerate(image_files)])
       
        # for idx,img_file in enumerate(image_files):
        #     img_path = os.path.join(class_path, img_file)
        #     image_to_graph(image_path=img_path,
        #                    label=label,
        #                    label_name=class_folder,
        #                    apply_transforms=apply_transform,
        #                    output_path=f"{graph_dataset_dir}/{label}_{idx}.pt")

        print(f"Contructed {len(image_files)} graphs  for Class #{label}: {class_folder} \n",flush=True)




def image_to_graph(image_path, label,label_name,connectivity,apply_transforms, output_path, use_image_feats, k=0.04, threshold=0.01):
    """
    Build a PyTorch graph from an image based on Harris corner detection.

    Args:
        image_path (str): Path to the image.
        k (float): Harris detector free parameter for detecting corners.
        threshold (float): Threshold for detecting strong corners.
        connectivity (str): Type of graph edges. Options: '4-connectivity', '8-connectivity'.

    Returns:
        torch_geometric.data.Data: PyTorch geometric data object.
    """

    # Step 1: Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")
    image = cv2.resize(image, (128, 128))

    height, width = image.shape

    # Step 2: Apply Harris corner detection
    harris_response = cv2.cornerHarris(image, blockSize=3, ksize=3, k=k)

    # Step 3: Apply threshold to extract strong corners
    corners = np.zeros_like(harris_response, dtype=np.uint8)
    corners[harris_response > threshold * harris_response.max()] = 1

    # Step 4: Extract (x, y) positions of corner points
    corner_positions = np.argwhere(corners == 1)  # Get row, col indices
    corner_indices = {tuple(pos): idx for idx, pos in enumerate(corner_positions)}
    node_features = corner_positions  # Node features are the (x, y) positions
    label = torch.tensor([label], dtype=torch.long)
    # Step 5: Create edges based on neighbor connectivity
    edges = []
    for pos in corner_positions:
        i, j = pos

        # Neighbor coordinate offsets for chosen connectivity
        neighbors = []
        if connectivity =='8-connectivity':
            neighbors = [
                (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
                (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)
            ]
        elif connectivity =='4-connectivity':
            neighbors = [
                (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)
            ]
        else:
            raise ValueError(f"Invalid connectivity '{connectivity}'. Choose '4-connectivity' or '8-connectivity'.")

        # Process each neighbor
        for ni, nj in neighbors:
            if 0 <= ni < height and 0 <= nj < width and corners[ni, nj] == 1:  # Within bounds and valid corner
                edges.append((corner_indices[tuple(pos)], corner_indices[(ni, nj)]))

    # Convert edges to PyTorch tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty((2, 0), dtype=torch.long)

    # Convert corner positions to PyTorch tensor as node features
    x = torch.tensor(node_features, dtype=torch.float)

    # foundation_model = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
    # feature_extractor = torch.nn.Sequential(*list(foundation_model.features.children()))
    # feature_extractor.eval()
    image = Image.open(image_path).convert('RGB')

    if apply_transforms:
        transform_pipeline = transform(type_data="train")
        img = transform_pipeline(image).unsqueeze(dim=0)
    else:
        transform_pipeline = transform(type_data="test")
        img = transform_pipeline(image).unsqueeze(dim=0)


    # with torch.no_grad():
    #     features = feature_extractor(img)

    # img_features = torch.flatten(features, start_dim=1)



    # Return PyTorch geometric Data object
    if use_image_feats==True:
            data = Data(x=x, edge_index=edge_index, y=label, image_features=img,label_name=label_name)
    else:
        data = Data(x=x, edge_index=edge_index, y=label,label_name=label_name)
    torch.save(data, output_path)

    return data