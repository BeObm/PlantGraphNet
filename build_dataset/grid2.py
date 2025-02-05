from concurrent.futures import ProcessPoolExecutor
from threading import Lock
import os
from tqdm import tqdm
import torch
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

lock = Lock()  # Lock for thread-safe I/O operations


def build_dataset(dataset_path, output_path, nb_per_class=200, apply_transform=True):
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff')  # File extension constant
    dataset = []
    class_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    def process_image(img_path, label, class_folder, graph_counter):
        # Process an individual image
        graph_data = image_to_graph(img_path, label, apply_transform)

        # Lock for printing and plotting to avoid concurrency issues
        with lock:
            if graph_counter <= 2:
                print(f"Graph {graph_counter} for Class #{label} ({class_folder}): {graph_data} \n")
                plot_image_with_nodes(
                    img_path,
                    graph_data,
                    f"{config['param']['result_folder']}/ImageAndGraph/{label}/{graph_counter}"
                )
        return graph_data

    def process_class_data(class_folder, label):
        class_path = os.path.join(dataset_path, class_folder)
        image_files = shuffle_dataset([
            f for f in os.listdir(class_path) if f.lower().endswith(IMAGE_EXTENSIONS)
        ])
        if nb_per_class > 0:
            image_files = image_files[:nb_per_class]

        # Parallel processing of images in the class
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(lambda img_file: process_image(
                    os.path.join(class_path, img_file), label, class_folder, image_files.index(img_file) + 1),
                             image_files )
            )
        return results

    # Iterate over classes and process them
    for label, class_folder in enumerate(tqdm(class_folders, desc="Processing classes")):
        class_results = process_class_data(class_folder, label)
        dataset.extend(class_results)  # Aggregate results from parallel processing

    # Save the dataset after processing
    torch.save(dataset, output_path)


