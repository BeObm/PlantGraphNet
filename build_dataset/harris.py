from skimage import color
from skimage.feature import corner_harris, corner_peaks
from torch_geometric.data import Data
from torchvision import transforms
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from utils import *
import os
from tqdm import tqdm


def image_to_graph(img_path, label):
    img = io.imread(img_path)
    # Check if the image is grayscale
    if len(img.shape) != 2:
        img = color.rgb2gray(img)

    img_tensor = transforms.ToTensor()(img)
    coords = corner_peaks(corner_harris(img_tensor[0].numpy()), min_distance=5)

    rgb_values = [img_tensor[:, int(y), int(x)].numpy() for x, y in coords]

    features = [np.append(coord, rgb) for coord, rgb in zip(coords, rgb_values)]
    features=np.array(features)
    x = torch.tensor(features, dtype=torch.float)
    print(f"X shape: {x[1]}")
    foundation_model = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
    feature_extractor = torch.nn.Sequential(*list(foundation_model.features.children()))
    feature_extractor.eval()
    image = Image.open(img_path).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(image)

    img_features = torch.flatten(features, start_dim=1)
    edge_index = torch.combinations(torch.arange(x.size(0), dtype=torch.long)).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=label, img_features=img_features)

    return data


# Build the PyTorch Geometric dataset
def build_dataset(dataset_path, output_path):
    dataset = []
    class_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for label, class_folder in enumerate(class_folders):
        pbar = tqdm(len(class_folders))
        pbar.set_description(f"Contructing graph data for label # {class_folder}... ")
        class_path = os.path.join(dataset_path, class_folder)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))]
        a = 1
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            data = image_to_graph(img_path, label)
            dataset.append(data)
            if a < 5:
                plot_image_with_nodes(img_path, data, f"{config['param']['result_folder']}/ImageAndGraph/{label}/{a}")
                a += 1
        pbar.set_description(f"Contructed graph {len(image_files)} graphs  for label # {label} ")
        pbar.update(1)
    torch.save(dataset, output_path)

