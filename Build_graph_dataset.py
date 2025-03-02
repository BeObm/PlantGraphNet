import argparse
from model import *
import importlib
from datetime import datetime
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
from skimage.morphology import skeletonize
from scipy.spatial import Delaunay, Voronoi
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.utils import grid
from build_dataset.build_dataset_utils import *




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
            
       
        print(f"Contructed {len(image_files)} graphs  for Class #{label}: {class_folder} \n")




def image_to_graph(img_path, label,label_name,node_detector,apply_transforms=True, output_path="data/graph_data.pt", use_image_feats=False):
    
    graph_constructor_obj = importlib.import_module(f"build_dataset.build_dataset_utils")
    graph_constructor = getattr(graph_constructor_obj, node_detector,label)
    
    data = graph_constructor(img_path,label)
    y = torch.tensor([label], dtype=torch.long)
    if use_image_feats==True:
         data.image_features=img2.unsqueeze(dim=0),
         data.label_name=label_name
    else:
        data.label_name=label_name
    torch.save(data, output_path)




if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()

    parser.add_argument("--type_node_detector", default="grid_graph", type=str, help="define how to detect nodes", choices=["grid_graph", "superpixel_graph", "keypoint_graph", "region_adjacency_graph", "delaunay_graph", "feature_map_graph","mesh3d_graph", "voronoi_graph", ]) 
    parser.add_argument("--apply_transform", default=True, type=bool, help="apply transform", choices=[True, False])
    parser.add_argument("--images_per_class", type=int, default=0, help="number of images to use for training/test per class; 0 means all")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--connectivity", type=str, default="4-connectivity", help="connectivity", choices=["4-connectivity", "8-connectivity"])
    parser.add_argument("--use_image_feats", default=False, type=bool, help="use input  image features as graph feature or not")

    args = parser.parse_args()


    create_config_file(args.type_node_detector)
    
    start_time = datetime.now()
    print(f" {'#'*10}  Creating training graph datasets...")
    build_dataset(dataset_path="dataset/images/train",
                                     args=args,
                                     type_dataset="train",
                                     apply_transform=True)
    print(f" {'#'*10}  Creating validation graph datasets...")
    build_dataset(dataset_path="dataset/images/val",
                                     args=args,
                                     type_dataset="val",
                                     apply_transform=False)
    print(f" {'#'*10}   Creating testing graph datasets...")
    build_dataset(dataset_path="dataset/images/test",
                                     args=args,
                                     type_dataset="test",
                                     apply_transform=False)

   
    print("Graph datasets created successfully in ", datetime.now() - start_time)